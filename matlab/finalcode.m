% Step 1: Load and preprocess the face images
imageDir = 'C:\Users\sshou\Desktop\matlab\dataset'; % Specify the directory containing face images
imageFiles = dir(fullfile(imageDir, '*.jpeg'));
numImages = length(imageFiles);
imageSize = [455, 455]; % Adjust the size according to your images
% Initialize variables to store training data
trainingData = zeros(prod(imageSize), numImages);
% Load and preprocess training images
for i = 1:numImages
    image = imread(fullfile(imageDir, imageFiles(i).name));
    % Resize the image to the desired size
    image = imresize(image, imageSize);
    % Convert to grayscale
    if size(image, 3) > 1
        image = rgb2gray(image);
    end
    % Convert to a column vector and store in trainingData
    trainingData(:, i) = image(:);
end
% Step 2: Compute the eigenfaces from the training data
meanFace = mean(trainingData, 2);
zeroMeanData = trainingData - meanFace;
[eigenfaces, eigenvalues] = pca(zeroMeanData');
% Step 3: Project each training face onto the eigenfaces
projectedData = eigenfaces' * zeroMeanData;
% Step 5: Prepare a new face for recognition (similar preprocessing as training images)
% Step 6: Project the new face onto the eigenfaces
newFace = imread('testrkb.jpeg'); % Load the new face
newFace = imresize(newFace, imageSize);
if size(newFace, 3) > 1
    newFace = rgb2gray(newFace);
end
newFaceVector = double(newFace(:)) - meanFace;
newFaceProjection = eigenfaces' * newFaceVector;

% Step 7: Compare the new face with the training data and determine the identity
% Compute the distances
distances = sum((projectedData - newFaceProjection).^2, 1);

% Find the index of the minimum distance
[~, recognizedPersonIndex] = min(distances);

% Map the index to a person (assuming each person has 9 images)
recognizedPerson = ['Person ' char('A' + floor((recognizedPersonIndex - 1) / 9))];

% Display the recognized person
fprintf('Recognized Person: %s\n', recognizedPerson);