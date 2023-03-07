# PDF-Viewer

Pdf Viewer is an application that allows users to view and edit PDFs for a more comfortable experience.

This app is programmed using pure computer vision, first by reading the PDF and then converting each page to an image, which can then be modified as desired.


The app provides the following features:

    Changing the colors of the text and background.
    Changing the colors to eye-comfort colors.
    Marking the page and scrolling using the eye.
    Taking a screenshot with the eye.

For the eye scrolling and taking screenshots with an eye, only CascadeClassifier, a machine learning algorithm, is used to get the indices for eyes and head.
