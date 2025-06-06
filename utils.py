class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {}
        self.model = model
        self.decay = decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.model.state_dict()[name].clone()