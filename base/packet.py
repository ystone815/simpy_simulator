class Packet:
    """
    A generic packet/transaction object for simulation.
    It carries attributes that describe a transaction.
    """
    def __init__(self,
                 id,
                 type,          # e.g., 'read', 'write', 'command', 'response'
                 source_id=None,
                 destination_id=None,
                 address=None,
                 size=None,     # Size in bytes or blocks
                 data=None,     # Payload for write/response data
                 timestamp=None, # When the packet was created/sent
                 **kwargs):     # For additional, specific attributes

        self.id = id
        self.type = type
        self.source_id = source_id
        self.destination_id = destination_id
        self.address = address
        self.size = size
        self.data = data
        self.timestamp = timestamp
        
        # Store any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        """Get attribute value with default"""
        return getattr(self, key, default)

    def __repr__(self):
        return (f"Packet(id={self.id}, type='{self.type}', src={self.source_id}, "
                f"dst={self.destination_id}, addr={self.address}, size={self.size}, "
                f"data_len={len(self.data) if self.data else 0}, ts={self.timestamp})")

    def __str__(self):
        return self.__repr__()
