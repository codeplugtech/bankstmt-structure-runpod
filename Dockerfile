RUN apt-get update && apt-get install -y poppler-utils


# Python dependencies

COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --ignore-installed --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#      It is reccommended to specify the version of Python when running your code.


# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py