class Memory:
    def __init__(self, id, text, template, losses=None, description=None):
        self.id = id
        self.text = text
        self.template = template
        self.losses = losses if losses is not None else []
        self.description = description
        self.embedding = None
    
    def __repr__(self):
        return f"<Memory {self.id} - {self.text[:30]}...>"
    
    def to_json(self, llm_memory):
        return {
            "id": self.id,
            "description": self.description,
            "token": llm_memory.memory_token.format(id=self.id),
            "text": self.text,
            "template": self.template,
            "embedding": self.embedding,
            "losses": self.losses
        }
