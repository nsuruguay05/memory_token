import pickle
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from .memory import Memory

DEVICE = "cuda"

class LLMWithMemory:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", load_in_4bit=True):
        """ Initializes the LLMWithMemory class.
        Args:
            model_id (str): The HuggingFace identifier for the pre-trained model to be loaded.
            load_in_4bit (bool): Whether to load the model in 4-bit quantization using BitsAndBytes.
        """
        # BitsAndBytes configuration for 4-bit quantization
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        # Initialize memory-related attributes
        self.memories = []
        self.memory_token = "<MEMORY>"
        self.selected_memory = None

        # Add the memory token to the tokenizer and model
        self.tokenizer.add_tokens([self.memory_token])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.memory_token_id = self.tokenizer.convert_tokens_to_ids([self.memory_token])[0]
        # Save the initial embedding to always start from the same point (for fair comparison)
        self.init_embedding = self.model.get_input_embeddings().weight.data[self.tokenizer.convert_tokens_to_ids([self.memory_token])]
        
        # Freeze all model parameters except the embedding layer
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.get_input_embeddings().weight.requires_grad = True

    def _train_embedding(self, text_to_remember, memory=None, epochs=3000, lr=5, warmup_iters=None, early_stopping=True, silenced=False):
        """ Trains the memory token embedding on the provided text.
        Args:
            text_to_remember (str): The text to train the memory token on.
            memory (Memory, optional): The Memory object to associate with this training. Defaults to None.
            epochs (int): Number of training epochs. Defaults to 3000.
            lr (float): Learning rate for the optimizer. Defaults to 5.
            warmup_iters (int): Number of warmup iterations for the learning rate scheduler. If None, no scheduler is applied. Defaults to None.
            early_stopping (bool): Whether to stop training early if the output matches the expected output. Defaults to True.
            silenced (bool): If True, suppresses progress bar output. Defaults to False.
        """
        # Initialize input and expected output
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs = self.tokenizer(text_to_remember, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        expected_output = inputs.input_ids.clone()
        expected_output[0, :-1] = inputs.input_ids[0, 1:] # Shift input ids to the left
        expected_output[0, -1] = self.tokenizer.eos_token_id # Set the last token to EOS

        # Initialize optimizer and scheduler
        optimizer = torch.optim.SGD([self.model.get_input_embeddings().weight], lr=lr)
        if warmup_iters is not None:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=2e-1, end_factor=lr, total_iters=warmup_iters)

        # Iterate through epochs
        pbar = tqdm(range(epochs)) if not silenced else range(epochs)
        for epoch in pbar:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            token_logits = self.model(**inputs).logits

            # Check for early stopping condition
            if early_stopping and expected_output[0].equal(token_logits.argmax(dim=-1)[0]):
                print(f"EARLY STOPPING - EPOCH {epoch}")
                break

            # Compute loss and backpropagate
            loss = loss_fn(token_logits.view(-1, token_logits.size(-1)), expected_output.view(-1))
            loss.backward()

            # Zero out gradients for all rows except the new embedding's row, ensuring only the memory token is updated
            with torch.no_grad():
                grad = self.model.get_input_embeddings().weight.grad
                mask = torch.zeros_like(grad)
                mask[self.memory_token_id] = 1
                self.model.get_input_embeddings().weight.grad = grad * mask

            # Step the optimizer and scheduler
            optimizer.step()
            if warmup_iters is not None:
                scheduler.step()

            # Record loss in memory if provided and update progress bar description
            if memory:
                memory.losses.append(loss.cpu().item())
            if not silenced:
                pbar.set_description(f"Epoch: {epoch} - Loss: {loss.item()}")

    def _set_memory_embedding(self, embedding):
        """ Sets the embedding for the memory token.
        Args:
            embedding (torch.Tensor): The embedding to set for the memory token.
        """
        self.model.get_input_embeddings().weight.data[self.tokenizer.convert_tokens_to_ids([self.memory_token])] = embedding

    def select_memory(self, memory_id):
        """ Selects the active memory by its ID.
        Args:
            memory_id (int): The ID of the memory to select.
        """
        self.selected_memory = self.memories[memory_id]
        self._set_memory_embedding(self.selected_memory.embedding)

    def add_memory(self, text, template, embedding=None, losses=None, description=None, **train_kwargs):
        """ Adds a new memory to the model.
        Args:
            text (str): The text to remember.
            template (str): The template to format the memory text.
            embedding (torch.Tensor, optional): Precomputed embedding for the memory token. If None, it will be trained from scratch. Defaults to None.
            losses (list, optional): List of losses to initialize the memory with. Defaults to None.
            description (str, optional): Description of the memory. Defaults to None.
        Returns:
            Memory: The newly created Memory object.
        """
        # Create a new Memory object and add it to the memories list
        new_memory = Memory(len(self.memories), text, template, losses=losses, description=description)
        new_memory.embedding = self.init_embedding if embedding is None else embedding
        self.memories.append(new_memory)
        self.select_memory(len(self.memories) - 1)

        # If no embedding is provided, train the embedding on the text
        if embedding is None:
            text_to_remember = new_memory.template.format(memory_token=self.memory_token, text=new_memory.text)
            self._train_embedding(text_to_remember, memory=new_memory, **train_kwargs)
            new_memory.embedding = self.model.get_input_embeddings().weight.data[self.tokenizer.convert_tokens_to_ids([self.memory_token])]

        return new_memory

    def add_memory_bulk(self, texts, templates, embeddings, losses, descriptions):
        """ Adds multiple already trained memories to the model in bulk.
        Args:
            texts (list of str): List of texts to remember.
            templates (list of str): List of templates to format the memory texts.
            embeddings (list of torch.Tensor): List of precomputed embeddings for the memory tokens.
            losses (list of list, optional): List of lists containing losses for each memory.
            descriptions (list of str, optional): List of descriptions for each memory.
        """
        new_memories = []
        new_indices = list(range(len(self.memories), len(self.memories)+len(texts)))
        for idx, text, embedding, template, loss, description in zip(new_indices, texts, embeddings, templates, losses, descriptions):
            new_memory = Memory(idx, text, template, losses=loss, description=description)
            new_memory.embedding = embedding
            new_memories.append(new_memory)
        
        self.memories.extend(new_memories)
    
    def save_memories(self, path):
        """ Saves all memories using pickle.
        Args:
            path (str): Path to save the file.
        """
        json_memories = [memory.to_json(self) for memory in self.memories]
        with open(path, 'wb') as f:
            pickle.dump(json_memories, f)
  
    def load_memories(self, path):
        """ Loads memories from a pickle file. 
        Args:
            path (str): Path to the file containing saved memories.
        """
        with open(path, 'rb') as f:
            loaded_memories = pickle.load(f)

        texts, templates, embeddings, losses, descriptions = [], [], [], [], []
        for memory in loaded_memories:
            texts.append(memory['text'])
            templates.append(memory['template'])
            embeddings.append(memory['embedding'])
            losses.append(memory['losses'] if 'losses' in memory else [])
            descriptions.append(memory['description'] if 'description' in memory else None)

        self.add_memory_bulk(texts, templates, embeddings, losses, descriptions)

    def generate(self, text, max_tok=2500):
        """ Generates text based on the provided input text using greedy decoding.
        Args:
            text (str): The input text to generate from.
            max_tok (int): The maximum number of tokens to generate.
        """
        # Ensure the memory token is set in the model's embedding
        if self.selected_memory is not None:
            self._set_memory_embedding(self.selected_memory.embedding)
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to('cuda')
        
        # Greedy decoding
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=False,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.memory_token_id]] # Remove memory id from generation
            )
        
        # Decode the result
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    def evaluate(self):
        """ Evaluates the model's performance on the currently selected memory.
        Returns:
            float: The accuracy of the model's output compared to the expected output.
        """
        if self.selected_memory is None:
            raise ValueError("No memory selected for evaluation.")
        
        # Ensure the memory token is set in the model's embedding
        self._set_memory_embedding(self.selected_memory.embedding)

        # Tokenize the memory text
        input = self.tokenizer("<MEMORY>" + self.selected_memory.text, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        # Get the model's logits for the input
        token_logits = self.model(**input).logits

        # Prepare the expected output
        expected_output = input.input_ids.clone()
        expected_output[0, :-1] = input.input_ids[0, 1:] # Shift input ids to the left
        expected_output[0, -1] = self.tokenizer.eos_token_id # Set the last token to EOS

        # Check perfect match to avoid potential issues with floating point precision
        if expected_output[0].equal(token_logits.argmax(dim=-1)[0]):
            accuracy = 1.0
        else:
            # Calculate accuracy
            accuracy = (expected_output == token_logits.argmax(dim=-1)).float().mean().item()
        
        return accuracy
