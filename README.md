# Sign-Language-Recognition
Sign Language Recognition System.

# Method: 
The static sign language data for our project was in the form of images. We trained a Convolutional Neural Network (CNN)
to identify the signs represented by each of these images. The dynamic sign language dataset we used was collected by a 
LeapMotion Controller (LMC) and was in the form of (x, y, z) coordinates of each joint of each hand collected every few 
milliseconds. We feature engineered this data to get useful relative motion data which was then trained on classical 
classification models to identify the specific sign pertaining to each LMC input.

# Applications: 
Our proposed system will help the deaf and hard-of-hearing communicate better with members of the community. 
For example, there have been incidents where those who are deaf have had trouble communicating with first responders when in need.

Another application is to enable the deaf and hard-of-hearing equal access to video consultations, whether in a professional
context or while trying to communicate with their healthcare providers via telehealth. Instead of using basic chat, these 
advancements would allow the hearing-impaired access to effective video communication.

#Performance: 
The proposed model for the still images is able to identify the static signs with an accuracy of 94.33%. Based on our 
analysis of the dynamic signs, we realized the need to identify if the sign is a one-handed or two-handed sign first, 
and then identify the sign itself. The final model we propose for the dynamic signs is capable of identifying the one-handed 
signs with an accuracy of 88.9% and the two-handed signs with an accuracy of 79.0%.
