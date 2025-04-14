FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install Spark dependencies
RUN apt-get update && apt-get install -y \
    curl \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
RUN curl -fsSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz | \
    tar -xz -C /opt/ && \
    ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# Install pyspark
RUN pip install --upgrade pip && pip install pyspark onnx onnxruntime

# Set working directory
WORKDIR /app

# Default to bash
CMD ["bash"]
