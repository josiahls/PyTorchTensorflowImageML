from pytorch_tensorflow_image_ml.utils.config import Config

if __name__ == '__main__':
    config = Config()

    # Envs to run
    config.add_argument('--environment', type=str, default='FetchReach-v2',
                        help='Define the environment to test.')