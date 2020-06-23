A basic means of training models saved with tensorflow. Build the model using the ./build_model.py script and then use the model with the ./UseModel.py script.
The input data is a path to a directory of the following structure.

Source_Directory
                |
                |
                |
                data
                |   |
                |   1.npy
                |   2.npy
                .   .
                .   .
                .   n.npy
                |  
                label
                    |
                    1.npy
                    2.npy
                    .
                    .
                    n.npy
                    
                    
 Where the path to Source_Directory is the input to the ./UseModel.py script.
 The ith input datum is Source_Directroy/data/i.npy and the ith label is Source_Directory/label/i.npy
