install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

test:
	pytest

docker-build:
	docker build -t rag-api .

docker-run:
	docker run -p 8000:8000 --env-file .env rag-api

pdf:
	python generate_pdf.py

clean:
	rm -rf vectorstore documents/*.pdf
