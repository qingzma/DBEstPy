NOTICE:
--------------------------------------------------------------------------------------------------------------------
This repository is no longer maintained,  a new version is at https://github.com/qingzma/DBEstClient, and will be released soon.
--------------------------------------------------------------------------------------------------------------------


DBEst Repository
========================

This project implements the Approximate Query Processing engine (AQP) of DBEst.
DBEst is a model-based AQP engine using regression models and density estimator.

Currently DBEst supports various aggregate funcitons, including COUNT, SUM, AVG, PERCENTILE, VARIANCE, STDDEV, MIN, MAX, etc.
Group By is also supported.

The main function is located in dbest/dbestclient.py

v2.0 RoadMap
---------------
1. Enable multi-thread training\\
2. Enable multi-thread prediction, especially group by\\
3. Transfer to Java
4. DBEst over spark
