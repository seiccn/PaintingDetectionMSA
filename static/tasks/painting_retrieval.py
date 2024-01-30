"""
Module containing functions to perform Painting Retrieval.
"""
import sys

from tasks.image_processing import automatic_brightness_and_contrast
from tasks.painting_rectification import rectify_painting
from models.painting import Painting

import pandas as pd
import cv2
import numpy as np
import os
import time
import ntpath


def create_paintings_db(db_path, data_path):
    """Creates a list of all paintings in the DB and their info.

    Parameters
    ----------
    db_path: str
        path of the directory containing all painting files.
    data_path: str
        path of the '.csv' file containing all paintings info.

    Returns
    -------
    list
        list of `Painting` objects, describing all the paintings that populate
        the DB.
    """
    paintings_db = []
    try:
        df_painting_data = pd.read_csv(ntpath.realpath(data_path))
    except UnicodeDecodeError:
        sys.exit(f"Error in the data file format: {data_path}")

    db_path = ntpath.realpath(db_path)
    print("DB path"+str(db_path))
    orb = cv2.ORB_create()
    for subdir, dirs, files in os.walk(db_path):
        print("Files")
        print(files)
        
        for painting_file in files:
            print("painting_file"+str(painting_file))
            image = cv2.imread(os.path.join(db_path, painting_file))

            painting_info = df_painting_data.loc[df_painting_data['Image'] == painting_file].iloc[0]
            title = painting_info['Title']
            author = painting_info['Author']
            room = painting_info['Room']
            print("Title"+str(title))
            print("Author"+str(author))
            print("Room"+str(room))
            # create ORB keypoints and descriptors for each painting in the DB
            src_kp, src_des = orb.detectAndCompute(image, None)

            painting = Painting(
                image,
                title,
                author,
                room,
                painting_file,
                keypoints=src_kp,
                descriptors=src_des
            )
            print(painting)
            paintings_db.append(painting)  
    return paintings_db


def match_features_orb(src_img, dst_img, max_matches=50):
    """Find the matches between two images.

    Find the matches between two images using ORB to find Keypoints.

    Parameters
    ----------
    src_img: ndarray
        source image
    dst_img: Painting
        destination image
    max_matches: int
        maximum number of the best (lower distance) matches found that we consider
    Returns
    -------
    float
        Returns the average distance value considering only the best (lower distance)
        `max_matches` matches.

    Notes
    -----
    For details visit
    - https://docs.opencv.org/3.4.0/d3/da1/classcv_1_1BFMatcher.html#ac6418c6f87e0e12a88979ea57980c020
    """
    orb = cv2.ORB_create()

    src_kp, src_des = orb.detectAndCompute(src_img, None)
    # show_image("src_kp", cv2.drawKeypoints(src_img, src_kp, None, color=(0, 255, 0), flags=0), wait_key=False)

    dst_kp, dst_des = dst_img.keypoints, dst_img.descriptors
    # stard adding
    
    print("src_des type:", src_des.dtype)
    print("src_des shape:", src_des.shape)

    # show_image("dst_kp", cv2.drawKeypoints(dst_img, dst_kp, None, color=(0, 255, 0), flags=0), wait_key=False)

    # Find matches between the features in the source image and the destination
    # image (i.e. painting)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(src_des, dst_des)
    matches_value = np.inf
    if len(matches) > 0:
        matches_value = np.mean([i.distance for i in sorted(matches, key=lambda x: x.distance)[:max_matches]])

    # Show matches
    # draw_params = dict(singlePointColor=None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # matches_img = cv2.drawMatches(src_img, src_kp, dst_img.image, dst_kp, matches, None, **draw_params)
    # show_image("matches_img", matches_img, wait_key=False)
    # cv2.waitKey(0)

    return matches_value


def histo_matching(src_img, dst_img):
    """Histogram Comparison of two images.

    Parameters
    ----------
    src_img: ndarray
        first image to compare
    dst_img: ndarray
        second image to compare

    Returns
    -------
    float
        Returns the value of the Histogram Comparison of two images using the
        Intersection method.

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    - https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
    - https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
    - https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga6ca1876785483836f72a77ced8ea759a
    """

    # Convert image from BGR to HSV
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)

    # Set number of bins for hue and saturation
    hue_bins = 50
    sat_bins = 60
    hist_size = [hue_bins, sat_bins]

    # Set ranges for hue and saturation
    hue_range = [0, 180]
    sat_range = [0, 256]
    hist_range = hue_range + sat_range

    # Use the 0-th and 1-st channels of the histograms for the comparison
    channels = [0, 1]

    # Create histograms
    src_hist = cv2.calcHist(
        [src_img],
        channels,
        None,
        hist_size,
        hist_range,
        accumulate=False
    )
    cv2.normalize(src_hist, src_hist, 0, 1, cv2.NORM_MINMAX)

    dst_hist = cv2.calcHist(
        [dst_img],
        channels,
        None,
        hist_size,
        hist_range,
        accumulate=False
    )
    cv2.normalize(dst_hist, dst_hist, 0, 1, cv2.NORM_MINMAX)
    hist_comparison = cv2.compareHist(src_hist, dst_hist, cv2.HISTCMP_INTERSECT)
    return hist_comparison


def painting_db_lookup(img, paintings_db, generator, show_image, print_next_step, print_time, max_matches=40,
                       threshold=0.92, match_db_image=False, histo_mode=False):
    """Lookup the DB for a specific painting using ORB or histogram matching.

    Parameters
    ----------
    img: ndarray
        the input image.
    paintings_db: list
        list of all `Painting` object that populate the DB.
    generator: generator
        generator used to print useful information during processing
    show_image: function
        function used to show image of the intermediate results
    print_next_step:function
        function used to print info about current processing step
    print_time: function
        function used to print info about execution time
    max_matches: int
        maximum number of the best (lower distance) matches found that we consider
    threshold: float
        rejects matches if the ratio between the best and the second-best match is
        below the `threshold`. Credits:
         https://stackoverflow.com/questions/17967950/improve-matching-of-feature-points-with-opencv
    match_db_image: bool
        define if:
        - rectify `img` one time using aspect ratio (True)
        - rectify `img` for every painting in `paintings_db`
    histo_mode: bool
        indicate which method use for the matching:
        - True = Histogram Matching.
        - False = ORB matching.

    Returns
    -------
    tuple[ndarray, bool]
        Returns:
        - a NumPy array of tuple (id, mean_distance), where id is the
        identification number of the current painting of the DB and
        mean_distance is the mean distance between `img` and the current
        painting of the DB. The array is sorted in ascending order of distance.
        - a boolean representing if a good match was found or not


    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
    - https://numpy.org/doc/1.18/reference/generated/numpy.sort.html
    """

    dtype = [('painting_id', int), ('val_matches', float)]
    matches_rank = np.array([], dtype=dtype)


    rectified_img = img
    # print(rectified_img.shape)
    print("before loop")
    for id, painting_obj in enumerate(paintings_db):
        painting_img = painting_obj.image
        # show_image('image_db', painting, wait_key=False)
        print("before match db image")
        if match_db_image:
            print("in match_db_image")
            # Step 15: Rectify Painting
            # ----------------------------
            print_next_step(generator, "Rectify Painting")
            start_time = time.time()
            corners = np.float32([
                [0, 0],
                [img.shape[1] - 1, 0],
                [img.shape[1] - 1, img.shape[0] - 1],
                [0, img.shape[0] - 1]
            ])
            rectified_img = rectify_painting(img, corners, painting_img)
            print_time(start_time)
            # show_image('image_rectified', rectified_img, wait_key=True)

            # Step AUTO-ADJUST: Adjust automatically brightness and contrast of the image
            # ----------------------------
            print_next_step(generator, "Adjust brightness and contrast")
            start_time = time.time()
            rectified_img, alpha, beta = automatic_brightness_and_contrast(rectified_img)
            # print(f"\talpha: {alpha}")
            # print(f"\tbeta: {beta}")
            print_time(start_time)
            # show_image('auto_adjusted', rectified_img)

        if not histo_mode:
            # Step 16: Match features using ORB
            # ----------------------------
            print_next_step(generator, "Match features using ORB")
            start_time = time.time()
            # max_distance = 40
            match_size = match_features_orb(rectified_img, painting_obj, max_matches)
            print_time(start_time)
        else:
            # Step 17: Match features using HISTOGRAMS
            # ----------------------------
            print_next_step(generator, "Match features using HISTOGRAMS")
            start_time = time.time()
            match_size = histo_matching(rectified_img, painting_img)
            print_time(start_time)

        matches_rank = np.append(matches_rank, np.array([(id, match_size)], dtype=dtype))
    print("Matches Rank")
    print(matches_rank)

    matches_rank = np.sort(matches_rank, order='val_matches')

    good_match_found = False
    print("Size is: "+ str(len(matches_rank)))
    if matches_rank.size > 0:
        if histo_mode:
            matches_rank = np.flip(matches_rank)
            if matches_rank[0][1] > 0:
                good_match_found = True
        else:
            if matches_rank[0][1] < np.inf and matches_rank[0][1] < matches_rank[1][1] * threshold:
                good_match_found = True

    return matches_rank, good_match_found


def retrieve_paintings(paintings_detected, paintings_db, generator, show_image, print_next_step, print_time,
                       match_db_image=False, histo_mode=False, scale_factor=1.):
    """Match each detected painting to the paintings DB.

    Parameters
    ----------
    paintings_detected: list
        a list containing one `Painting` object for each
        painting detected in the input image.
    paintings_db: list
        a list containing one `Painting` object for each
        painting in the DB.
    generator: generator
        generator function used to take track of the current step number
        and print useful information during processing.
    show_image: function
        function used to show image of the intermediate results
    print_next_step:function
        function used to print info about current processing step
    print_time: function
        function used to print info about execution time
    match_db_image: bool
        define if:
        - False: rectify each painting one time using a calculated aspect ratio
        - True: rectify each painting to match the aspect ration of every painting in `paintings_db`
    histo_mode: bool
        indicates whether to perform a Histogram Matching in the case ORB
        does not produce any match.
    scale_factor: float
        scale factor for which the original image was scaled

    Returns
    -------
    None
        add painting DB info (e.g. title, author, room, etc.) to the paintings
        in the `paintings_detected` list
    """
    paintings_retieved = []
    for i, painting in enumerate(paintings_detected):
        print('\n# Processing painting #%d/%d' % (i + 1, len(paintings_detected)))

        sub_img = painting.image

        sub_img = cv2.resize(
            sub_img,
            (int(sub_img.shape[1] / scale_factor), int(sub_img.shape[0] / scale_factor)),
            cv2.INTER_CUBIC
        )

        # Step AUTO-ADJUST: Adjust automatically brightness and contrast of the image
        # ----------------------------
        print_next_step(generator, "Adjust brightness and contrast")
        start_time = time.time()
        img_auto_adjusted, alpha, beta = automatic_brightness_and_contrast(sub_img)
        # print(f"\talpha: {alpha}")
        # print(f"\tbeta: {beta}")
        print_time(start_time)
        show_image('auto_adjusted_image', img_auto_adjusted)

        sub_img = img_auto_adjusted

        # Step 14: Painting DB lookup
        # ----------------------------
        threshold = 0.92
        print_next_step(generator, "Painting DB lookup")
        max_matches = 30  # 40
        matches_rank, good_match_found = painting_db_lookup(
            sub_img,
            paintings_db,
            generator=generator,
            show_image=show_image,
            print_next_step=print_next_step,
            print_time=print_time,
            threshold=threshold,
            max_matches=max_matches,
            match_db_image=match_db_image,
        )
       
        if not good_match_found and histo_mode:
            if matches_rank[0][1] == np.inf:
                matches_rank, good_match_found = painting_db_lookup(
                    sub_img,
                    paintings_db,
                    generator=generator,
                    show_image=show_image,
                    print_next_step=print_next_step,
                    print_time=print_time,
                    max_matches=max_matches,
                    match_db_image=match_db_image,
                    histo_mode=histo_mode
                )
            else:
                good_match_found = True
        if good_match_found:
            painting_id = matches_rank[0][0]
            # Manage case when I find duplicated painting in the current video frame
            if painting_id is not None and painting_id not in paintings_retieved:
                paintings_retieved.append(painting_id)
                recognized_painting = paintings_db[painting_id]
                painting.title = recognized_painting.title
                painting.author = recognized_painting.author
                painting.room = recognized_painting.room
                painting.filename = recognized_painting.filename
                show_image("DB_image_predicted", recognized_painting.image)

        print('\n# Painting #%d/%d information:' % (i + 1, len(paintings_detected)))
        if painting.title is not None:
            print("\t{:10s} {}".format("title:", painting.title))
            print("\t{:10s} {}".format("author:", painting.author))
            print("\t{:10s} {}".format("room:", painting.room))
            print("\t{:10s} {}".format("filename:", painting.filename))
        else:
            print("\t-- No DB match --")

        # cv2.waitKey(0)

    # return paintings_detected
