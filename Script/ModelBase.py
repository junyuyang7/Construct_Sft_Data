

class BaseModel:
    def __init__(self):
        pass

    def generate(self):
        pass


class TopicModel(BaseModel):
    def __init__(self):
        super.__init__(self, TopicModel)

    def generate(self):
        return super().generate()
    

class AskModel(BaseModel):
    def __init__(self):
        super.__init__(self, AskModel)

    def generate(self):
        return super().generate()
    

class AnswerModel(BaseModel):
    def __init__(self):
        super().__init__(self, AnswerModel)

    def generate(self):
        return super().generate()
