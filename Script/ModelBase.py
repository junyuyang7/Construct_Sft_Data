

class BaseModel:
    def __init__(self):
        pass

    def generate(self):
        pass


class TopicModel(BaseModel):
    def __init__(self):
        super().__init__()

    def generate(self, prompt):
        return "调用Topic Model"
    

class AskModel(BaseModel):
    def __init__(self):
        super().__init__()

    def generate(self, prompt):
        return "调用Ask Model"
    

class AnswerModel(BaseModel):
    def __init__(self):
        super().__init__()

    def generate(self, prompt):
        return "调用Answer Model"
    

class JudgeModel(BaseModel):
    def __init__(self):
        super().__init__()

    def generate(self, prompt):
        return "调用Judge Model"
