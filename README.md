
Kyle Hoffpauir, Adam Kolides
Dr. Simon
Big Project Proposal
Abstract
This work aims to train 3 lightweight machine learning frameworks (Tensorflow Lite,
Lasagne, and MXNet) to evaluate how well they can perform with a specific dataset. We are
specifically looking at how well they function with regards to the Internet of Things and Edge
Computing. We will evaluate our machine learning models on cpu usage, runtime, processor
time, and accuracy to determine which lightweight machine learning library and model
combination yields the best results.
Introduction
The Internet of Things is an expanding market rooted in embedded systems.
Applications are leveraging the new power of microcomputers combined with machine learning
to perform what is being called Edge Computing (Bazai). By placing sensors on the edge, data
can be collected and sent to the cloud for processing and powerful results from a small internet
connected device. This power can be further harnessed through an emerging paradigm of Edge
Intelligence (Deng), placing the computing power of machine learning on these small IoT
devices can create systems that are much more robust and advanced than traditional
applications. However, this edge intelligence paradigm requires advancements of existing
technology in the form of lightweight, portable systems. The market has thus grown with a new
type of Artificial Intelligence system in the form of lightweight machine learning frameworks.
These frameworks exist on the edge as sensors, using pre-trained models to classify incoming
data in place before sending it to the cloud. This increases the overall power of the network and
eases the strain on the cloud services hosting the edge intelligence model. The following work
focuses on 3 lightweight machine learning frameworks: Tensorflow Lite, Lasagne, and MXNet.
We aim to train these applications on the same dataset, a collection of temperatures and
evaluate them (Data found here:
https://www.kaggle.com/datasets/atulanandjha/temperature-readings-iot-devices). The
applications will be trained using a variety of techniques, such as regression, decision trees, and
clustering, and will be evaluated based on their cpu usage, accuracy, and speed. The ideal
lightweight ML frameworks can produce a model which can accurately predict future data using
the least amount of computational force the fastest.
Related Work
● Hafizur Rahman, Md. Iftekhar Hussain,
A light-weight dynamic ontology for Internet of Things using machine learning technique,
ICT Express,
Volume 7, Issue 3,
2021,
Pages 355-360,
ISSN 2405-9595,
https://doi.org/10.1016/j.icte.2020.12.002.
(https://www.sciencedirect.com/science/article/pii/S2405959520304902)
● C. Wang, G. Papadimitriou, M. Kiran, A. Mandal and E. Deelman, "Identifying Execution
Anomalies for Data Intensive Workflows Using Lightweight ML Techniques," 2020 IEEE
High Performance Extreme Computing Conference (HPEC), 2020, pp. 1-7, doi:
10.1109/HPEC43674.2020.9286139.
https://ieeexplore.ieee.org/abstract/document/9286139/keywords#keywords
● G.P. Sajeev, M.P. Sebastian,
Building semi-intelligent web cache systems with lightweight machine learning
techniques,
Computers & Electrical Engineering,
Volume 39, Issue 4,
2013,
Pages 1174-1191,
ISSN 0045-7906,
https://doi.org/10.1016/j.compeleceng.2013.02.005.
(https://www.sciencedirect.com/science/article/pii/S0045790613000311)
● Bzai, Jamal, et al. "Machine Learning-Enabled Internet of Things (IoT): Data,
Applications, and Industry Perspective." Electronics 11.17 (2022): 2676.
● Deng, Shuiguang, et al. "Edge intelligence: The confluence of edge computing and
artificial intelligence." IEEE Internet of Things Journal 7.8 (2020): 7457-7469.
● https://towardsdatascience.com/tiny-machine-learning-the-next-ai-revolution-495c26463
868
● https://www.tensorflow.org/lite
● https://www.seeedstudio.com/blog/2021/06/14/everything-about-tinyml-basics-courses-pr
ojects-more/
● https://www.strong.io/blog/edge-machine-learning-from-poc-to-real-world-ai-applications
● https://mxnet.apache.org/versions/1.9.1/
● https://lasagne.readthedocs.io/en/latest/
● https://www.kaggle.com/code/andreshg/automl-libraries-comparison
Proposed Work
In order to evaluate the performance of all of the lightweight ML frameworks, we will
create a python application to do machine learning on a dataset, namely the temperature
dataset. We aim to create a python application that will create a set of models for each
framework. Each will implement regression, decision trees, and clustering. We will train the
models and test them on this dataset. This portion of the application will represent the
pre-trained cloud system.
Once these models have been trained to the best of their capabilities, the lightweight ML
frameworks will be implemented in a separate python object which will represent the portion of
the frameworks that exists on the edge. The edge systems only have access to the pre-trained
model and the input data, it will need to use this pre-trained model to classify the incoming data
accurately. Each model will receive an influx of new mock “sensor data”, matching in style what
they have been trained on. These models will then need to parse the data and use the
pre-trained models they have access to from the cloud system in order to classify the data and
produce a prediction. The lightweight frameworks will also send their results to the cloud in
order to update the model in the cloud.
As the mock edge devices run their machine learning, we will be taking metrics
concerning cpu usage, runtime, and accuracy in order to get an accurate, numerical
representation of the effectiveness of the different systems. These tests will be run a set amount
of times and averaged in order to provide a more accurate metric.
As an element of the project proposal, we are considering attempting to run these
frameworks on a microcomputer such a raspberry pi. However, this would come with the
caveats of securing a raspberry pi to work on and such may not be a viable option.
Evaluation
We will evaluate each machine learning model to see which framework/model
combination has the best run time and accuracy. More specifically, the model with the smallest
run time but highest accuracy level will be the goal. We will also be looking at memory usage,
processor time, processor usage, and other similar metrics.
Timeline
Week 1 (10/31 - 11/7) Lightweight ML investigation & software design
Week 2 (11/7 - 11/14) Implementing software (creating & training models)
Week 3 (11/14 - 11/21) Implementing software (running & evaluating frameworks)
Week 4 (11/21 - 11/28) Presentation/Paper construction
Team Composition and Responsibilities
Adam - Choose which ML techniques to use and evaluate, create graphs/charts, related work
Kyle - Find Lightweight ML algorithms and datasets, create graphics (if applicable). Check
kaggle for a good dataset
Both - Write and run code to obtain performance metrics, write paper/presentation (exact
sections will be decided at time of writing
