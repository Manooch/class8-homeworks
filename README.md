# class8-homeworks

Loading breast cancer dataset from scikit-learn, plotting and make prediction with supervised machine learning algorithms:

KNeighbors
GaussianNB
DecisionTree

How to run breastcanceranalysis.py using Docker
  - Copy these files to your local repository:
		breastcanceranalysis.py
		Dockerfile

  - Run the script in this format:
		docker build -t <image_name> <path>
		docker run -it -v /${PWD}:/${PWD} -w /${PWD} <image_name> <dataset name>

  - Figures are saved in Figures folder

When breastcanceranalysis.py is run:
  - First, it plots Histogram 2D and Histogram grouped for the provided dataset (with matplotlib)
  - Second, it plots Histogram 2D and Scatter for the dataset loaded from sklearn.datasets
  - Third, it class the above algorithms to do some predictions