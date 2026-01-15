train:
	papermill train.ipynb train.out.ipynb

build:
	docker build -t cleptoninja .

run:
	docker run -it cleptoninja
