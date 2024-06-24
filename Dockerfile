# Select base image (can be ubuntu, python, shiny etc)
FROM python:3.12-slim

# Create user name and home directory variables. 
# The variables are later used as $USER and $HOME. 
ENV USER=username
ENV HOME=/home/$USER

# Add user to system
RUN useradd -m -u 1000 $USER

# Set working directory (this is where the code should go)
WORKDIR $HOME/kg

# Update system and install dependencies.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common

# Copy code and start script (this will place the files in home/username/)
COPY requirements.txt $HOME/kg/requirements.txt
COPY webapp $HOME/kg/webapp/
COPY data/ $HOME/kg/data/
COPY start-script.sh $HOME/kg/start-script.sh

RUN pip install --no-cache-dir -r requirements.txt \
    && chmod +x start-script.sh \
    && chown -R $USER:$USER $HOME \
    && rm -rf /var/lib/apt/lists/*

USER $USER
EXPOSE 8501

ENTRYPOINT ["./start-script.sh"]