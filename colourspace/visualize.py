import matplotlib.pyplot as plt
def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    # input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('hhh',  input_image)
    # cv2.waitKey(0)

    fig = plt.figure()
    plt.imshow(input_image)
    plt.show(block=False)
    pass
