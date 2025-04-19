from ts.torch_handler.base_handler import BaseHandler
from torchtext.data.utils import get_tokenizer
import torch
import json

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.model = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.tokenizer = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details
        self.model = torch.jit.load('model_scripted.pt')
        self.tokenizer = get_tokenizer('basic_english')
        try:
            with open('vocab_to_idx.json', 'r') as f:
                self.vocab_to_idx = json.load(f)
        except FileNotFoundError:
            print("Error: vocab_to_idx.json not found")

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        
        if self.vocab_to_idx is not None:
            tokens = []
            for token in self.tokenizer(data.get("data")):
                tokens.append(self.vocab_to_idx.get(token, 0))
            return torch.tensor([tokens])
        else:
            print("Error: vocab_to_idx.json not found")
            return None


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
    

if __name__ == "__main__":
    handler = ModelHandler()
    handler.initialize(None)
    print(handler.handle({'data': "this was the best movie i have ever seen"}, None))