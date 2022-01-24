import cv2
import sys
import random
import numpy as np

"""
This is a frame differencing motion detector with an adaptive threshold to 
select the most intense pixel changes. The largest changes are very often
 related to the motion of the foreground object (or objects). But sometimes 
 that assumption does not work. To minimize that error, you can localize 
 subsequent searches. To ensure the first frame is correctly initialized, 
 you can focus the search manually (mouse cursor), if the auto-detection 
 is not working. If you are familiar with classical CV techniques for 
 image processing (blurring, morphologies, centroids, contours, masking, etc.),
  a combination of those techniques can refine/filter the frame differencing 
  to minimize false positives. There is no clear cut formula for which combination 
  and settings work, so I had to test many versions. That, unfortunately, is classical CV: 
  a lot of trial and error looking for settings that work across many situations.
"""
cam = cv2.VideoCapture('boat.mp4')

NEW_SIZE_FACTOR = 0.4
OBJ_DIM = (20, 30) # (height, width)
CROSSHAIR_DIM = (15, 15)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

NUM_PARTICLES = 200
PARTICLE_SIGMA = np.min([OBJ_DIM])//4 #Particle filter shift per generation
DISTRIBUTION_SIGMA = 0.5

def make_crosshair(img, top_left, bot_right, ch_color, captured): 
    obj_h, obj_w = CROSSHAIR_DIM
    center_y, center_x = img.shape[0]//2, img.shape[1]//2
    
    img = cv2.rectangle(img, top_left, bot_right, ch_color, 1)
    img = cv2.line(img, (center_x, img.shape[0]* 1//3), (center_x, center_y - obj_h//2), ch_color, 1)
    img = cv2.line(img, (center_x, center_y + obj_h//2), (center_x, img.shape[0] * 2 // 3), ch_color, 1)
    img = cv2.line(img, (img.shape[1] * 1//3, center_y), (center_x - obj_w//2, center_y), ch_color, 1)
    img = cv2.line(img, (center_x + obj_w//2, center_y), (img.shape[1] * 2//3, center_y), ch_color, 1)
    return img


def mark_target(img, center_xy, ch_color, captured):
    obj_h, obj_w = OBJ_DIM[0], OBJ_DIM[1]
    center_x, center_y = int(center_xy[0]), int(center_xy[1])

    t1_x = int(center_xy[0] - obj_w//2)
    t1_y = int(center_xy[1] - obj_h//2)
    br_x = int(center_xy[0] + obj_w//2)
    br_y = int(center_xy[1] + obj_h//2)

    transformations = ( 
        ((center_x, 0), (center_x, center_y - obj_h//2)),
        ((center_x, center_y + obj_h//2), (center_x, img.shape[0])),
        ((0, center_y), (center_x - obj_w//2, center_y)), 
        ((center_x + obj_w//2, center_y), (img.shape[1], center_y))
    ) #in order

    img = cv2.rectangle(img, (t1_x, t1_y), (br_x, br_y), ch_color, 1)
    for first, second in transformations: 
        img = cv2.line(img, first, second, ch_color, 1)
    return img

def imgisEmpty():
    if img is None: 
        cam.release()
        print("Image was empty.")
        sys.exit(0)
        return True
    return False
### MAIN PARTS 

captured = False 
img_patch = np.zeros(OBJ_DIM) #Changes to 3d image later! 

particles_xy, particles_scores, particles_patches = [], [], []
ret, img = cam.read() #read first frame to init variables
imgisEmpty() 
obj_h, obj_w = OBJ_DIM[0], OBJ_DIM[1]

img_color = cv2.resize(img, (int(img.shape[1] * NEW_SIZE_FACTOR), int(img.shape[0] * NEW_SIZE_FACTOR)))
img_h, img_w, _ = img_color.shape
top_left_x, top_left_y = top_left = (img_w//2 - obj_w//2, img_h//2 - obj_h//2)
bot_right_x, bot_right_y = bot_right = (img_w//2 + obj_w//2, img_h//2 + obj_h//2)
while True: 
    _, img = cam.read() 
    imgisEmpty()

    img_color = cv2.resize(img, (int(img.shape[1] * NEW_SIZE_FACTOR), int(img.shape[0] * NEW_SIZE_FACTOR)))
    img_color_clean = img_color.copy() 
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)


    if captured: 
        key = cv2.waitKey(10) & 0xFF 
    else: 
        key = cv2.waitKey(25) & 0xFF
    if key == 27: #esc 
        break
    
    if key == ord('c') or key == ord('C'):
        captured = True
        img_patch = img_color_clean[top_left_y:bot_right_y, top_left_x:bot_right_x]
        particles_xy = np.zeros((NUM_PARTICLES, 2))
        particles_xy[:,:] = [img_w//2, img_h//2]
    elif key == ord('d') or key == ord('D'):
        captured = False
        img_patch = np.zeros(img_patch.shape)

    if captured: 
        # Introduce noise for the next generation 
        for i, p in enumerate(particles_xy):
            if i == 0: 
                continue
            
            p[0] += np.random.normal(0, PARTICLE_SIGMA)
            p[1] += np.random.normal(0, PARTICLE_SIGMA)


            ##### TODO:  This might be wrong idk tho
            # p[0] = min(max(obj_w//2, p[0]), img_w - obj_w//2)
            # p[1] = min(max(obj_h//2, p[1]), img_h - obj_h//2)     
            p[0] = min(max(obj_w//2, p[0]), img_w - obj_w//2)
            p[1] = min(max(obj_h//2, p[1]), img_h - obj_h//2)   

        #Display particles
        for p in particles_xy:
            img_color = cv2.circle(img_color, (int(p[0]), int(p[1])), 1, GREEN, -1) 

        #Get patches for each particle
        particles_patches = []
        for p in particles_xy:
            patch_top_left_x, patch_top_left_y = top_left  = (int(p[0] - obj_w//2), int(p[1] - obj_h//2))
            patch_bot_right_x, patch_bot_right_y = (int(p[0] + obj_w//2), int(p[1] + obj_h//2))
            temp_patch = img_color_clean[patch_top_left_y:patch_bot_right_y, patch_top_left_x:patch_bot_right_x]
            particles_patches.append(temp_patch)
        
        # Compare each patch with a model patch
        model_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
        model_patch = cv2.GaussianBlur(model_patch, (3, 3), 0)
        particles_scores = []
        for p in particles_patches:
            temp_patch = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
            temp_patch = cv2.GaussianBlur(temp_patch, (3, 3), 0)
            mse = np.mean((model_patch - temp_patch) ** 2)
            particles_scores.append(mse)

        #Convert to a probability
        particles_scores = np.array(particles_scores)
        paricles_scores = 1.0/(2 * np.pi * DISTRIBUTION_SIGMA) * np.exp(-particles_scores/(2 * DISTRIBUTION_SIGMA))
        particles_scores = particles_scores/np.sum(particles_scores)

        # Resample
        new_pxy_idx = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES - 1, p=particles_scores, replace=True)
        best_idx = np.where(particles_scores == np.max(particles_scores))[0][0]
        best_xy = particles_xy[best_idx]
        new_set = particles_xy[new_pxy_idx]
        particles_xy = np.vstack((best_xy, new_set))

        #display best_xy/ Mark target
        img_color = mark_target(img_color, best_xy, RED, 1)


        #update model patch
        img_patch = particles_patches[best_idx]

    img_color = make_crosshair(img_color, top_left, (bot_right_x, bot_right_y), GREEN, 1)
    cv2.imshow("Object Tracker", img_color)

cam.release()
cv2.destroyAllWindows()

            