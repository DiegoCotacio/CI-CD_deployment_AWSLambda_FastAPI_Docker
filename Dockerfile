FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

COPY models ./models

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --target .

COPY app.py .
#COPY app_utils.py .

CMD [ "app.handler" ]



