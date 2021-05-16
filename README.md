## ❓ What is our project:
	During the past year, lots of controversial events have happened, and no matter which side you are on, there’s no denying that there is lots of tension in the air constantly. This is a problem, as it means that people are much more likely for someone to bring up a sensitive topic or hit a nerve with someone else. Although many people are doing their best to keep calm in stressful times such as right now, fights will inevitably break out, and in large cities that span long distances, it can be very difficult to keep all of the unrest under control. 
	As a solution to this problem, the Rescue_PI aims to achieve accurate detection of social unrest using image processing and deep learning and do it quickly and efficiently to cover large areas in a fast and timely manner. Rescue_PI achieves speed and efficiency using a drone as a platform. As it flies along, its camera scans the surrounding environment and processes the video feed through an onboard computer in real-time. If a fight is detected, the integrated speaker on Rescue_PI plays a loud sound to alert passersby about the conflict going on, so that they can help de-escalate the situation.

## 💻 How we made the code for our project:
       Our code consisted of two models trained in two networks, TensorFlow and PyTorch. The TensorFlow model was intended for a try-it-yourself section of our website so that the Judges and anyone else interested would be able to see how we were able to detect the accidents/tensions in the way that we do. The PyTorch model was intended for the actual deployment on the Pi 4 + Drone System in our prototype, and we would detect and report the accidents based on how our PyTorch model predicted the live footage from the drone.

## 🔧 Hardware Components used:
           - PC with Nvidia 1080
           - MacBook Pro
           - Prototype
           - Raspberry Pi 4
           - Raspberry Pi Camera
           - Bluetooth Speaker 
           - Intel Movidius Stick
           - Various craft materials
           - DJI Mavic 2 Pro Drone


## 💻 Software Components Used:
       Webpage:
              - IDE: Jetbrains WebStorm
              - HTML
              - CSS
              - Javascript
        Model Training and Deployment (Prototype):
              - IDE: Jetbrains PyCharm
              - TensorFlow
              - PyTorch
              - OpenCV
              - Imutils
              - NumPy
              - Threading
              - Pygame
              - ONNX
              - Scikit-Learn
              - Matplotlib
              - PIL
              - Teachable Machine (only for deployment code in TensorFlow JS)
              - Animations and Presentation:
              - Blender
              - Google Slides

## 😢 Problems Faced? It wouldn’t be a hackathon without them!:
1. We could not find a suitable dataset that accurately depicts what we wanted to train and deploy in this hackathon. 
2. Deployment of an Image Classifier on a website was new to us, so we encountered problems on our path to do this. 
3. Figuring out how to properly mount the Raspberry Pi 4 onto the drone so that we could attain accurate videos and classification. 
4. Deployment of a proper model on the drone that would not be affected by the flight of the drone.

## ✅ The Problems We Solved!:
1. Created our own dataset by taking lots of pictures at the angles that we needed to detect. 
2. Learned how to use TensorFlow JS from Teachable Machine. 
3. Mounted the Raspberry Pi on the drone with household materials, balsa wood. tape, fishing line, and hot glue. 
4. We tailored our model to the drone’s POV so that it would be as accurate as possible. 

## 📆 Plans for the Future?:
     - Automate the drone entirely without human intervention.
     - Add detection for house fires, forest fires, natural disasters, and car crashes to help save more lives.
     - Alert police authorities when a fight is detected.
     - Detect if fights end with injuries, and if there are injuries, contact emergency medical services

## 🔗 Links:
       Website: https://rescuepi.rescuepi.repl.co/
       Youtube Video: 
       Github: https://github.com/RescuePi/Rescue_Pi_Code.git

Slideshow:https://docs.google.com/presentation/d/1HA29vRFW38c4qvKotIBeh_lra1jF28U5s8ywq324WLk/edit?usp=sharing
