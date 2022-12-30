from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.models import Perceptron

def main(data, modelName,plotName,eta,epochs):
    df_xor = pd.DataFrame(data)

    X, y = prepare_data(df_xor)

    model_xor = Perceptron(eta=eta, epochs=epochs)
    model_xor.fit(X, y)

    _ = model_xor.total_loss()

    model_xor.save(filename=modelName, model_dir="model")
    save_plot(df_xor,model_xor,filename=plotName)
    

if __name__=='__main__':

    XOR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [1,0,0,1]
    }

    ETA= 0.3
    EPOCHS=10

    main(data = XOR,modelName = "xor.model",plotName='xor.png',eta=ETA,epochs=EPOCHS)

