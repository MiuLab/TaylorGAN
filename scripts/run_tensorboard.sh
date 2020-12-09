if [ -f .env ]
then
  export $(cat .env | xargs)
  echo "Set environment variable from .env"
fi

tensorboard --logdir $TENSORBOARD_LOGDIR --port $TENSORBOARD_PORT
