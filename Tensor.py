import tensorflow as tf

mnist = tf.keras.datasets.mnist#loading mnist dataset.

(x_train, y_train), (x_test,y_test) = mnist.load_data()#loading MNist Data Set into tuples

x_train,  x_test = x_train/255.0, x_test /255.0 # converting data to float

model = tf.keras.models.Sequential([ #creating Sequential keras model.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy() #or each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions

print(predictions)

#convert logitcs into probablities.
tf.nn.softmax(predictions).numpy() # not sure how to get this value right now.

#The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#This loss is equal to the negative log probability of the true class: It is zero if the model is sure of the correct class.

#This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.log(1/10) ~= 2.3.

loss_fn(y_train[:1], predictions).numpy()

#compiles model.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


#The Model.fit method adjusts the model parameters to minimize the loss:

model.fit(x_train, y_train, epochs=5)

#evaluates model performance.
model.evaluate(x_test,  y_test, verbose=2)

model.summary()

model.save('saved_model')

# The above code is from the into to tensorflow, as an example DL model.
#This following line is run from command line, and converts this tensorflow model into a ONNX model. saving it as model.onnx
#python -m tf2onnx.convert --saved-model saved_model --output model.onnx  