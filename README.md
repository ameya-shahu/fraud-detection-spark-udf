# spark-udf-cnn-inference 

### Command to build docker image
```docker build -t spark-cnn-runtime .```


### Command to run spark job
```docker run --rm -it --gpus all -v $(pwd):/app spark-pytorch spark-submit /app/<SPARK-JOB FILE PATH>```