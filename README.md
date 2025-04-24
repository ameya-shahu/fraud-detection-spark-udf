# spark-udf-cnn-inference 

### Command to build docker image
```docker build -t spark-cnn-runtime .```


### Command to run spark job
```docker run --rm -it --gpus all -v $(pwd):/app spark-cnn-runtime spark-cnn-runtime /app/spark_infer.py <BACKEND TYPE>```

BACKEND TYPE can take two values - 
1. onnx
2. pytorch


To run inference on simple python use folloeing command
```docker run --rm -it --gpus all -v $(pwd):/app spark-cnn-runtime spark-cnn-runtime /app/python_infer.py```
