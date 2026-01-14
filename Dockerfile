FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

ENV APP_DIR='/app'
WORKDIR $APP_DIR

ADD requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --compile -r /tmp/requirements.txt && \
    rm -rf /tmp/* /var/tmp/*


ADD *.py ${APP_DIR}
ADD best_actor_critic_player.pt ${APP_DIR}

CMD python play.py
