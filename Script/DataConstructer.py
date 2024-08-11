from Script.ModelBase import AskModel, JudgeModel, AnswerModel, TopicModel

class DataConstructer:
    def __init__(self, topic_model: TopicModel, ask_model: AskModel, answer_model: AnswerModel, judge_model: JudgeModel):
        self.topic_model = topic_model
        self.ask_model = ask_model
        self.answer_model = answer_model
        self.judge_model = judge_model

    def get_query_from_topic(self, topic: str, prompt: str, context: str=None):
        reps = self.topic_model.generate(prompt)
        return reps + '  This is a query from topic'
        
    def get_query_from_answer(self, prompt: str, answer: str):
        reps = self.ask_model.generate(prompt)
        return reps + "  This is a query from answer"
        
    def get_answer_from_query(self, prompt: str, query: str):
        reps = self.answer_model.generate(prompt)
        return reps + "  This is a answer from query"

    def judge_dialog(self, prompt: str, dialog: str):
        reps = self.judge_model.generate(prompt)
        return 5

    def judge_answer(self, prompt: str, answer: str):
        reps = self.judge_model.generate(prompt)
        return 5

    def judge_query(self, prompt: str, query: str):
        reps = self.judge_model.generate(prompt)
        return 5

    def change_answer(self, prompt: str, answer: str):
        return "  a new answer"

    def change_query(self, prompt: str, query: str):
        return "  a new query"
    
