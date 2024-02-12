FROM python:3.11

RUN python -m pip install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR /usr/app

COPY ./support_chat ./support_chat
COPY ./tests ./tests
COPY ./data ./data

# note: README.md is required for Poetry
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN pip install .

EXPOSE 8502
ENTRYPOINT ["streamlit", "run", "support_chat/search.py", "--server.port=8080", "--server.address=0.0.0.0"]