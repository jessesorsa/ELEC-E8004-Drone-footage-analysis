from ML_model import predict
from luminance import luminance


def main():

    scaling_factor = 3

    print("Starting predict")
    corners_list = predict.predict()
    print("Bounding boxes aqcuired")

    print("Starting luminance")
    luminance.main(scaling_factor)
    print("Luminances done")

    print("Overlaying boxes")
    predict.add_bounding_boxes(corners_list, scaling_factor)


if __name__ == '__main__':
    main()
