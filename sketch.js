//Github: https://lunahmy.github.io/p5js-surveillance/. 
//Interactive data visualization using ml5.js handPose, bodyPose, and custom Neural Networks. 

//**DATA**
//This sketch mainly relies on data from the AI Global Surveillance (AIGS) Technology index published in 2018 and updated in 2022 by Steven Feldstein at the Carnegie Endowment for International Peace: https://data.mendeley.com/datasets/gjhf5y4xjp/4. 
//**LIMITATIONS** 
//Not comprehensive of all relevant technology, government surveillance uses, and applicable companies.
//Transparency bias: governments and many companies purposely hide their surveillance capabilities.
//Cetegorization bias: narrowed to 4 types
//Field research in some countries limited, anecdotal/circumstantial use excluded from index. 
//Instances of AI surveillance documented are not necessarily tied to harmful outcomes. 
//The market prediction data, updated April 2026, is supplemented by: https://www.marketresearchfuture.com/reports/ai-in-surveillance-market-39120#. 

//Frame rate = 60 fps

//**MODELS & ANALYSIS** 
//bodyPose (blazePose): 33 keypoints, each with x, y, and z 
//Eye detection: whether or not eyes are present in the screen. Toggles the automated motion of the data points.

//handPose: 21 keypoints, each with x, y, and z
//Both Hands:
//Wrist and Middle Finger MCP (metacarpophalangeal) KP Distances: relatively stable value to simulate depth calculation from camera. Maximum pixel distance threshold of one hand or both is mapped to background opacity and tracked KP opacity. 
//Hand Area: pixel space occupied by each hand, calculaed from a bounding box snippet with w * l. Minimum pixel area threshold of one hand or both toggles draw of country labels.
//Hand Count: Do not show correlation line when both hands are detected.
//Primary Hand:
//Centroid Location: also from the bounding box snippet, calculated from the bounding box with minX, maxX, minY, maxY. Whichever quadrant of the canvas the centroid is in highlights a corresponding category of AI surveillance used. 
//Index Finger Tip X-pos: values from both hands stored in an array every 60 frames, each frame compares position to determine positive/negative direction to label left/right, resets every second to reduce jitter. The canvas is mirrored so the labeling is opposite, left label toggles color coding mode revealing the regime types from the dataset. 
//Overall Detection Confidence: mapped to the stroke width of correlation line.

//Neural Network (Classification)
//The classifier NN uses this training dataset of 13,575 annotated gestures from extracted from children’s pointing, 2025: https://zenodo.org/records/16420298.
//The CSV is uploaded on github: https://github.com/lunahmy/pointing-gesture-dataset.
//Model training sketch: https://editor.p5js.org/lunahm/sketches/85cgciVT-. 
//When the primary hand is pointing, draw the bar chart of market projections instead. Otherwise, draw the scatter plot 

//Neural Network (Regression)
//Model training sketch: https://editor.p5js.org/lunahm/sketches/nxmqQWsKW and https://editor.p5js.org/lunahm/sketches/Bo7KIyNDB.  
//Prediction runs separately on each hand so they can play different notes, only trained on index finger tip y position. 
//Model is trained on U (up), C (center), and D (down) on the canvas, with U tied to max value in billions, C to median, and D to minimum. These values are mapped to midi notes and converted back to frequencies for the oscillators, and are also mapped to the volumes.

//Other mapped features:
//Total military expenditures of each country is mapped to the opacity of each data point. 
//The total number of types of AI surveillance used by each country is mapped to the radius of each data point. 

//See bottom of sketch for data variable descriptions and code references. 


let data = []; 
let x;
let y; 
let webcams = []; 
let webcam;
let webcamIndex = 0; //Change this to 1 to use an external camera  
let tileW;
let tileH; 
let tileCountX = 2;
let tileCountY = 2;
let tileCount = tileCountX * tileCountY;
let handPose; 
let hands = []; 
let tileActive = [];
let margin = 200; 
let marginBtm = 300; 
let radius = 5;
let colorGroups = {};  
let opacity; 
let tileMode = []; 
let loaded = []; 
let loadedMode = false; 
let rates = []; 
let maxRate; 
let dotColor; 
let handParts = ['wrist', 'middle_finger_mcp'];
let indexPrevX = 0;
let direction;
let distance; 
let palm; 
let confidence; 
let boundingBoxes = [];
let centroid = { x: 0, y: 0 };
let area; 
let handData = [];
let bgA = 255; 
let handA = 0; 
let maxDist; 
let distThres = 200;
let centroidTileVal; 
let centroidTile; 
let body;
let parts = [];
let stabLine = false; 
let angle; 
let eyeKeys = [1, 2, 3, 4, 5, 6]
let poses = [];
let tech; 
let repIndex;
let minRep;
let maxRep; 
let militaryExp; 
let minExp;
let maxExp; 
let minTech; 
let maxTech; 
let govSurv;
let minGov;
let maxGov; 
let demIndex; 
let minDem;
let maxDem; 
let showColors = false; 
let brain; 
let classifying = false; 
let state = "prediction";
let classIsModelLoaded = false;
let prediIsModelLoaded = false;
let currentPose; 
let pointing = false; 
let data2 = []; 
let minBil;
let maxBil;
let billions; 
let brain2;
let fingerTipIndex = 8; 
let notes = {
  U: 45.67, 
  C: 30.485, 
  D: 15.3,
}
let wave;
let predicting = false; 
let value; 
let scrollY; 
let op = 0;
let fadeAmount = 2; 
let lines = [];
let linesIndex = 0; 
let handArea = []; 
let minHandA = 0;
let handAreaThresh = 20000; 
let minHandReached = false; 
let csvData; 
let modelReady = false;
let wave2;
let predicting2 = false;
let centroidYes = false; 
let fontRegular; 
let fontBold;
let fontItalics;
let fontMedItalics;
let frames = 60; 
let motionTracking = [];
let directionTxt = true; 

function preload(){
  //Load ML5 Model
  handPose = ml5.handPose();  
  bodyPose = ml5.bodyPose("BlazePose");
  
  //Options for the neural network models 
  let options2 = {
    inputs: 1,
    outputs: ['billions'],
    task: "regression",
    debug: true,
  };

  let options = {
      task: 'classification',
      debug: false,
  };

  //Load the neural network mdodels  
  brain = ml5.neuralNetwork(options);
  brain2 = ml5.neuralNetwork(options2)
  
  //Load Datasets 
  data = loadTable("data/cleaned-global2.csv", "csv", "header"); 
  data2 = loadTable("data/market-data.csv", "csv", "header"); 
  
  //Load fonts 
  fontRegular = loadFont('firsNeue/TT Firs Neue Trial Regular.ttf');
  fontBold = loadFont('firsNeue/TT Firs Neue Trial Bold.ttf');
  fontItalics = loadFont('firsNeue/TT Firs Neue Trial Italic.ttf');
  fontMedItalics = loadFont('firsNeue/TT Firs Neue Trial Medium Italic.ttf');

  //Load text file for bottom banner 
  lines = loadStrings('messages.txt');

}

function setup(){
  createCanvas(1920, 1080);
  x = width/2;
  y = height/2; 
  angleMode(DEGREES);
  pixelDensity(1);
  
  //Initialize the webcam with the chosen index, start bodyPose and handPose models. Reference: https://editor.p5js.org/MingrenFu/sketches/p2YR1AMDU.   
  getVideoDevices(getVideo);
  
  //Initial y-position for fading scrolling text at the bottom 
  scrollY = height - 30;
  
  //Set initial font 
  textFont(fontRegular);
  textAlign(CENTER, CENTER);
  tileW = width / tileCountX;
  tileH = height / tileCountY;
  
  //Load model files for neural networks 
  let modelDetails = {
    model: "modelZ/model.json",
    metadata: "modelZ/model_meta.json",
    weights: "modelZ/model.weights.bin",
  };

  let modelDetails2 = {
    model: "model2/model.json",
    metadata: "model2/model_meta.json",
    weights: "model2/model.weights.bin",
  };
  brain.load(modelDetails, classModelLoaded);
  brain2.load(modelDetails2, predModelLoaded);
  
  //Initialize audio outputs 
  wave = new p5.Oscillator();
  wave.setType('sine');
  wave.start();
  wave.freq(440);
  wave.amp(0);
  wave2 = new p5.Oscillator();
  wave2.setType('sine');
  wave2.start();
  wave2.freq(440);
  wave2.amp(0);

  //Sort through data and store in variables  
  repIndex = data.getColumn("repIndex").map(parseFloat);
  minRep = min(repIndex); 
  maxRep = max(repIndex); 
  militaryExp = data.getColumn("militaryExp").map(parseFloat); 
  maxExp = max(militaryExp); 
  minExp = min(militaryExp); 
  tech = data.getColumn('totalTech')
  minTech = min(tech);
  maxTech = max(tech); 
  govSurv = data.getColumn("govSMSurv").map(parseFloat); 
  minGov = min(govSurv);
  maxGov = max(govSurv);
  demIndex = data.getColumn("demIndex").map(parseFloat); 
  minDem = min(demIndex);
  maxDem = max(demIndex);
  billions = data2.getColumn("billions").map(parseFloat); 
  minBil = min(billions);
  maxBil = max(billions);

}

//Had to add this for publishing on the web, added toggle directions text. 
function mousePressed() {
  userStartAudio();
  directionTxt = false;  
}

//Callbacks to check if models loaded  
function classModelLoaded(){
  console.log("classification model loaded");
  state = 'prediction'; 
  classIsModelLoaded = true;   
  modelReady = true;
}
function predModelLoaded(){
  console.log("prediction model loaded");
  state = 'prediction'; 
  predIsModelLoaded = true;   

}


//https://editor.p5js.org/MingrenFu/sketches/p2YR1AMDU 
function getVideoDevices(callback) {
  navigator.mediaDevices
    .enumerateDevices()
    .then((devices) => {
      const filtered = devices.filter((device) => device.kind === "videoinput");
      callback(filtered);
    })
    .catch((err) => console.log(err));
}

function getVideo(cams) {
  for (let cam of cams) {
    let index = cams.indexOf(cam);
    let constraints = {
      audio: false,
      video: {
      deviceId: {exact: cam.deviceId},
      },
    };
    webcams[index] = createCapture(constraints, () => {
      if (index === webcamIndex) {
        webcams[webcamIndex].size(width, height); //Start handPose and bodyPose here 
        handPose.detectStart(webcams[webcamIndex], gotHands);
        bodyPose.detectStart(webcams[webcamIndex], gotPoses);
        console.log("Detection started on webcams[" + index + "]");
      }
    });
    webcams[index].hide();
    console.log(`webcams[${index}] ${cam.label}`);
  }
}



function draw() {
  //Set up flipped canvas 
  push();
  translate(width, 0);
  scale(-1, 1);

  //Initialize all modes and their variables 
  loadedMode = false; 
  dotColor = color(255);
  maxDist = 0;
  stabLine = false; 
  angle = frameCount % 360;
  centroidTileVal = -1;
  confidence = 0;
  showColors = false; 
  minHandReached = false; 

  
  //Reset Tile States  
  for (let i = 0; i < tileCount; i++){
    tileActive[i] = false;
  }
  radius = 5;

  //Draw wbecam footage 
  if (webcams.length > 1) {
    image(webcams[webcamIndex], 0, 0, width, height);
  }else{
    image(webcams[webcamIndex], 0, 0, width, height);

  }

  //Draw grid 
  for (let i = 0; i < tileCountX; i++){
    for (let j = 0; j < tileCountY; j++){
      let sx = i * tileW;
      let sy = j * tileH;
      let index = i + j * tileCountX;
      stroke(0);
      noFill();
      rect(sx, sy, tileW, tileH);
    }
  }

  //Draw eyes keypoints 
  for (let h = 0; h < poses.length; h++){
    let pose = poses[h];    
    for (let k = 0; k < pose.keypoints.length; k++){
      let keypoint = pose.keypoints[k];
      if (!eyeKeys.includes(k)) continue; //Skip if index value is not includeed in array 
      fill(0, 255, 255); 
      noStroke();
      ellipse(pose.keypoints[k].x, pose.keypoints[k].y, 5);
      stabLine = true; //Stabilize the dot chart, data is observed  
    }
  }
  
  //Classify inputs if hands are detected, model is loaded and classification is not already running 
  if (state === "prediction" && hands.length > 0 && !classifying && modelReady && classIsModelLoaded) {
    classifying = true;
    let wristX = hands[0].keypoints[0].x; //get wrist x and y position of primary hand 
    let wristY = hands[0].keypoints[0].y;
    let rawDists = []; //use dist() to get pixel distance from wrist to every keypoint 
    for (let k = 1; k < hands[0].keypoints.length; k++) {
      rawDists.push(dist(wristX, wristY, hands[0].keypoints[k].x, hands[0].keypoints[k].y));
    }
    let maxD = max(rawDists);
    let inputs = rawDists.map(d => maxD > 0 ? d / maxD : 0); //Normalize distances to 0-1 because training dataset was normalized, not in pixels 
      // for (let k = 1; k < hands[0].keypoints.length; k++) {
      //   inputs.push(dist(wristX, wristY, hands[0].keypoints[k].x, hands[0].keypoints[k].y));
      // }
      brain.classify(inputs, gotResult);
    } else if (!tileActive[0]) {
      currentPose = "";
  }  
  
  //Predict first hand inputs if hands are detected, model is loaded and this prediction is not already running
    if (state === "prediction" && hands.length > 0 && predIsModelLoaded && !predicting && pointing) {
      predicting = true; 
      let inputPredict = [];
      let tipY = hands[0].keypoints[fingerTipIndex].y; //Get the index finger tip y-position and push to prediction array 
      inputPredict = [tipY];
      brain2.predict(inputPredict, gotResult2); //Call prediction 
    }
  
  //Predict second hand inputs if hands are detected, model is loaded and this prediction is not already running 
    if (state === "prediction" && hands.length > 1 && predIsModelLoaded && !predicting2 && pointing) {
      predicting2 = true; 
      let inputPredict = [];
      let tipY = hands[1].keypoints[fingerTipIndex].y;
      inputPredict = [tipY];
      brain2.predict(inputPredict, gotResult3); 
    }
  
  //Draw black background  
  background(0, 0, 0, bgA);
  
  //Draw the centroid  
  drawCentroid(centroid.x, centroid.y, 15);

  //Turn off first audio when no hands exist
  if(hands.length === 0){
    wave.amp(0, 0.3); 
  }
  
  //Turn off second audio when less than 2 hands exist 
  if (hands.length < 2) {
  wave2.amp(0, 0.3);  
  }

  
  //Create handData objects to draw shapes and text 
  for (let h = 0; h < hands.length; h++){
    let hand = hands[h];
        handData[h] = {
        confidence: hand.confidence,
        palm: "",
        direction: handData[h]?.direction || "right", //Make sure it returns undefined if no handData exists, default to "right"
        distance: 0,
        area: 0,
        centroid: { x: 0, y: 0 },
        indexPrevX: handData[h]?.indexPrevX || 0,
        centroidTile: 0,
    };
    
  
  //Set the confidence returned by the model to the confidence data value of each hand 
    confidence = hand.confidence; 

  //Get distance from camera to hand from distance between wrist and middle finger mcp keypoints. Compare distances between hands and map maxDistance to background color opacity. 
    let rx = hand.wrist.x;
    let ry = hand.wrist.y;
    let lx = hand.middle_finger_mcp.x;
    let ly = hand.middle_finger_mcp.y;
    handData[h].distance = dist(rx, ry, lx, ly); 
    maxDist = max(maxDist, handData[h].distance); //Find max() of hand distance values 
      if(maxDist >= distThres){
      bgA = map(maxDist, distThres, 300, 255, 0);
      handA = map(maxDist, distThres, 300, 0, 255);
    }else{
      bgA = 255; 
      handA = 0; 
    }

    
  //Get x-pos of index finger tip of both hands every 60 frames and add to motionTracking array, compare x-pos from every frame to the stored array value [hand[0].indexTipX, hand[1].indexTipX]. If the x-pos is positive 200px, label "left" because canvas is mirrored and same for "right." After 60 frames, update the saved array. Original method of checking frame-by-frame was too jittery.  
    let indexTipX = hand.index_finger_tip.x; 
    if (frameCount % 60 === 0) {
      motionTracking[h] = hand.index_finger_tip.x;
    }
    if (motionTracking[h] !== undefined) {
      let vx = indexTipX - motionTracking[h];
      if (vx > 200) {
        handData[h].direction = "left";
      } else if (vx < -150) {
        handData[h].direction = "right"; 
      }
    }
  //Draw rectangles at these handParts and draw a magenta line between them, with the opacity mapped to the distance variable above. 
    push();
    stroke(255, 0, 255, handA);
    noFill();
    line(rx, ry, lx, ly);
    pop();
    for (let p of handParts) {
      let keypoint = hand[p];
      stroke(255, 0, 255, handA);
      noFill();
      rect(keypoint.x, keypoint.y, 15);  
    }
    
    
  //Draw all detected keypoints with the opacity mapped to the distance variable above. 
  for (let k = 0; k < hand.keypoints.length; k++){
    let keypoint = hand.keypoints[k];
    fill(0, 255, 0, handA);
    noStroke();
    circle(keypoint.x, keypoint.y, 10);
    }
  }

  //Check which tiles centroid location falls in and activate that tile and set centroidTileVale to tile value for text. 
  if(hands.length > 0){
    for (let i = 0; i < tileCountX; i++){
      for (let j = 0; j < tileCountY; j++){
        let index = i + j * tileCountX;
        let sx = i * tileW;
        let sy = j * tileH;
        if (centroid.x > sx && centroid.x < sx + tileW && centroid.y > sy && centroid.y < sy + tileH){
         tileActive[index] = true;
        }
        if (tileActive[0] == true) {
          centroidTileVal = 0; 
        }
        if (tileActive[1] == true) {
          centroidTileVal = 1; 
        }
        if (tileActive[2] == true) {
          centroidTileVal = 2; 
        }
        if (tileActive[3] == true) {
          centroidTileVal = 3; 
        }
      }
    }
  }

  //Call draw bounding boxes 
  drawBoundingBox();
  

  //Draw text using average x and y position calculated from all keypoints
  for (let h = 0; h < hands.length; h++){
      let hand = hands[h];
      push();
      translate(width, 0);
      scale(-1, 1);
      if (hand.keypoints && hand.keypoints.length > 0) {
        let avgX = hand.keypoints.reduce((sum, point) => sum + point.x, 0) / hand.keypoints.length;
        let avgY = hand.keypoints.reduce((sum, point) => sum + point.y, 0) / hand.keypoints.length - margin;
        fill(255, 255, 255, 255);
        textSize(20);
        let gap = 15; 
        noStroke();
        textAlign(CENTER, TOP);
        let isPrimary = (h === 0);
        text(`Hand ${h + 1} Detected`, width - avgX, avgY);

        if(isPrimary){
          text("Distance: " + round(handData[h].distance) + "px", width-avgX, avgY + gap*2);
          text("Wave Direction: " + handData[h].direction, width-avgX, avgY + gap*4);
          text("Hand Confidence: " + round(handData[h].confidence, 2), width-avgX, avgY + gap*6);
          text("Hand Space: " + round(handData[h].area) + "px^2", width-avgX, avgY + gap*8);
          text("Tile Location: " + centroidTileVal, width-avgX, avgY + gap*10);
          if (state === "prediction") {
            text("Pointing: " + pointing, width-avgX, avgY + gap*12);
          }
        }else{
          text("Distance: " + round(handData[h].distance) + "px", width-avgX, avgY + gap*2);
          text("Hand Space: " + round(handData[h].area)  + "px^2", width-avgX, avgY + gap*4);
        }
      }
    pop();
  }
  pop();

  //Check the direction variable of the first hand in handData to activate categorical color coding when "left" 
  if (hands.length > 0) {
    let firstHand = handData[0];
    if (firstHand.direction === "left") {
      showColors = true;
    } else if (firstHand.direction === "right") {
      showColors = false;
    }
  }
  
  //Calculate or reset minHandA for handArea threshold calculation.
  if(hands.length > 0){
    minHandA = min(handArea);
  }else{
    minHandA = 0;
  }
  
  //Calculate if minimum hand area is below the threshold and reveal country labels if the threshold is reached/viewer is far back enough from the camera. 
  if(minHandA < handAreaThresh && handArea.length > 0){
    minHandReached = true; 
  }

  //Draw the scatter plot if pointing gesture is not detected, and only draw the correlation line when ony one hand is detected. If pointing gesture is detected in first hand, switch to drawing bars, and default to drawing the scatter plot.
  if(!pointing){
    if(hands.length == 1){
    drawLine();
    }
    drawCircles();
  }else if(pointing && hands.length > 0){
    drawBars();
  }else{
    if(hands.length == 1){
    drawLine();
    }
    drawCircles();
  }
  
  //Make sure no sounds are playing when the hand is not pointing 
  if (!pointing) {
  wave.amp(0, 0.2);
  }
  if (!pointing || hands.length < 2) {
    wave2.amp(0, 0.2);
  }
  
  //Draw labels for visible/activated categorical groupings when both hands are present and not pointing 
  if(hands.length > 1 && centroidTileVal >= 0 && !pointing){
    push();
    noStroke();
    fill(255);
    textSize(32);
    textAlign(CENTER, CENTER);
    text("Countries Using AI:", x, margin - 100); 
    if(centroidTileVal == 0){
      text("Smart/Safe City System", x, margin - 50);
    }
    if(centroidTileVal == 2){
      text("Smart Policing", x, margin - 50);
    }
    if(centroidTileVal == 1){
      text("Facial Recognition Systems", x, margin - 50);
    }

    if(centroidTileVal == 3){
      text("Social Media Surveillance", x, margin - 50);
    }
    pop();
  }
  
  //Reset backgrounds and tracking transparency when no hands are detected 
  if (hands.length === 0) {
    bgA = 255;
    handA = 0;
    maxDist = 0;
  }

  
  //Draw key for categorical color coding mode only if not pointing and color mode is active.
  if(showColors && !pointing){
    let spacing = 20; 
    let startX = width - margin - 90; 
    push();
    textAlign(LEFT, BOTTOM);
    textSize(20);
    noStroke();
    textFont(fontRegular);
    push();
    textAlign(CENTER);
    textFont(fontBold);
    fill(255);
    let miniMargin = 325; 
    text("Regime Type", width - margin, miniMargin); 
    pop();
    fill(0, 255, 0);
    ellipse(startX, 0 + miniMargin + spacing*2, spacing);
    text("Liberal Democracy", startX + spacing, 0 + miniMargin + 10 + spacing*2); 
    fill(255, 0, 255);
    ellipse(startX, 0 + miniMargin + spacing*4, spacing);
    text("Electoral Democracy", startX + spacing, 0 + miniMargin + 10 + spacing*4); 
    fill(0, 255, 255);
    ellipse(startX, 0 + miniMargin + spacing*6, spacing);
    text("Electoral Autocracy", startX + spacing, 0 + miniMargin + 10 + spacing*6); 
    fill(255, 0, 0);
    ellipse(startX, 0 + miniMargin + spacing*8, spacing);
    text("Closed Autocracy", startX + spacing, 0 + miniMargin + 10 + spacing*8); 
    pop(); 
  }
  
  //Directions only before mouse press 
  if(directionTxt){
    direcTxt();
  }
  
  //Draw the bottom animated text bar 
  context();
}


//Callback to return classifications fot the pointing gesture classification NN model 
function gotResult(results){
  if (results && results.length > 0) {
      currentPose = results[0].label; //P or N 
      confidence = results[0].confidence; //0-1
    
      if (currentPose == 'P' && confidence > 0.9) {
        pointing = true;
      } else if (currentPose == 'N' && confidence > 0.5) {
        pointing = false;
        wave.amp(0, 0.3);  
      }
      if(hands.length === 0){
        pointing = false;  
        wave.amp(0, 0.3); 
      }
  }
  classifying = false; //End classification 
}


//Callback to to return predictions from first hand's prediction NN model 
function gotResult2(results){
  if (!results || results.length === 0) {
    predicting = false;
    return;
  }
  if (!hands || hands.length === 0) {  
    predicting = false;
    return;
  }
  let val = results[0].value + 10; //Results[0].value raw value scaled to be more audible 
  let midiNote = map(val, minBil, maxBil, 36, 60); //Convert to MIDI note because db are so small 
  let freq = midiToFreq(midiNote); //midiToFreq() method to convert back to frequency 
  let tipY = hands[0].keypoints[fingerTipIndex].y; //Finger tip height mapped to volume 
  let vol = map(tipY, 0, height, 1, 0.8); 
  vol = constrain(vol, 0, 1); //So that it stays within range 
  wave.amp(vol, 0.1);
  wave.freq(freq, 0.1);
  predicting = false; //End prediction 
}


//Callback to return predictions from second hand's prediction NN model 
function gotResult3(results){
  if (!hands || !hands[1] || !hands[1].keypoints) {
    predicting2 = false;
    return;
  }
  if (!results || results.length === 0) {
    predicting2 = false;
    return;
  }
  if (!hands || hands.length === 0) {  
    predicting2 = false;
    return;
  }
  let val = results[0].value + 10; 
  let midiNote = map(val, minBil, maxBil, 36, 60);
  let freq = midiToFreq(midiNote);
  let tipY = hands[1].keypoints[fingerTipIndex].y;
  let vol = map(tipY, 0, height, 1, 0.8); 
  vol = constrain(vol, 0, 1);
  wave2.amp(vol, 0.1);
  wave2.freq(freq, 0.1);
  predicting2 = false;
}


//Callback for bodyPose model, returning eye keypoints  
function gotPoses(results){
  poses = results;
}


//Callback for handPose model, returning hand keypoints and bounding box. 
function gotHands(results){
  hands = results;
  
  boundingBoxes = hands.map(hand => {
    if (!hand.keypoints || hand.keypoints.length === 0) return null;
      let xCoords = hand.keypoints.map(p => p.x);
      let yCoords = hand.keypoints.map(p => p.y);
      let xMin = Math.min(...xCoords);
      let xMax = Math.max(...xCoords);
      let yMin = Math.min(...yCoords);
      let yMax = Math.max(...yCoords);  
    return {
      xMin: xMin,
      xMax: xMax,
      yMin: yMin,
      yMax: yMax,
      width: xMax - xMin,
      height: yMax - yMin
    };
  }).filter(box => box != null);
  
  if (boundingBoxes[0]) {
    const box = boundingBoxes[0]; 
    centroid.x = (box.xMin + box.xMax) / 2; //Find centroid values 
    centroid.y = (box.yMin + box.yMax) / 2;
  }
}


//DRAWING FUNCTIONS BELOW 
function drawBoundingBox(){
  handArea = [];
  boundingBoxes.forEach((box, i) => { //Can i turn this into (for b in boundingBox)??
    if (!box) return;
    noFill();
    stroke(255, 0, 0, handA);
    strokeWeight(2);
    rect(box.xMin, box.yMin, box.width, box.height);
    handData[i].area = box.width * box.height; 
    handArea[i] = handData[i].area;        
  });
}



function drawCentroid(x, y, r){
  if(hands.length>0){
    push();
    fill(255, 0, 0, 255);
    noStroke();
    ellipse(x, y, r);
    pop();
  }
}


//Found the slope using Excel trend line, mapped it to the min and max of the data and margin of chart. 
function drawLine() {
  let m = -0.1496;
  let b = 0.5529;
  let x1 = margin;
  let x2 = width - margin;
  let y1 = map(m * minGov + b, minDem, maxDem, height - marginBtm, margin);
  let y2 = map(m * maxGov + b, minDem, maxDem, height - marginBtm, margin);
  let lineA; 
  let lineStroke; 
  if(hands.length > 0){
    lineStroke = map(confidence, 0.99, 1, 0, 10);
    push();
    noStroke();
    fill(255);
    textAlign(LEFT);
    textSize(24);
    text("y = -0.1496x + 0.5529", x1 + margin + 10, 100);
    text("Detection Confidence: " + round(confidence, 2), x1 + margin + 10, 130);
    pop();

  }else{
    lineStroke = 0; //No line when no hands are detected 
  }
  push();
  stroke(255);
  noFill();
  strokeWeight(lineStroke);
  line(x1, y1, x2, y2);
  pop();
}


function drawCircles(){
  if(stabLine){
    speed = 0;
  }
  else{
    speed = 45;
  }

  for(let r = 0; r < data.getRowCount(); r++){
    let rowObject = data.getRow(r);
    let dotX = map(rowObject.getNum("govSMSurv"), minGov, maxGov, margin, width - margin);
    let dotY = map(rowObject.getNum("demIndex"), minDem, maxDem, height - marginBtm, margin); 
    let moveY = dotY + sin(angle + r * 30) * speed; 
    radius = map(rowObject.getNum("totalTech"), minTech, maxTech, 30, 60);
    opacity = map(rowObject.getNum("militaryExp"), minExp, maxExp, 100, 255);
    country = rowObject.getString("Country");
    if(showColors){
      if(rowObject.getString("regimeType") == "LD"){
        dotColor = color(0, 255, 0, opacity); 
      }
      else if(rowObject.getString("regimeType") == "CA"){
        dotColor = color(255, 0, 0, opacity); 
      }
      else if(rowObject.getString("regimeType") == "ED"){
        dotColor = color(255, 0, 255, opacity); 
      } 
      else if(rowObject.getString("regimeType") == "EA"){
        dotColor = color(0, 255, 255, opacity); 
      } 
      else{
        dotColor = color(255, 255, 255, opacity); 
      }
    }else{
      dotColor = color(255, 255, 255, opacity); 

    }

    if(hands.length > 1){
      let safe = rowObject.getString("safeCity");
      let facial = rowObject.getString("facialRec");
      let policing = rowObject.getString("smartPoli");
      let social = rowObject.getString("smSurv");

      if(centroidTileVal === 0 && safe !== "x") continue;
      if(centroidTileVal === 1 && facial !== "x") continue;
      if(centroidTileVal === 2 && policing !== "x") continue;
      if(centroidTileVal === 3 && social !== "x") continue;
    }
    push();
    fill(dotColor);
    noStroke();
    ellipse(dotX, moveY, radius); 
    pop();

    //If minimum threshold for hand area is reached, also draw country labels 
    if(minHandReached){
      drawCountries(country, dotX, moveY); 
    }  
  }  
  push();
  fill(255);
  textSize(24);
  textAlign(CENTER);
  push();
  translate(margin - 50, height / 2);
  rotate(270);    
  textFont(fontBold);
  text("Varieties of Democracy (V-Dem) Index", 0, 0);
  pop();
  push();
  translate(margin - 50, height / 2);
  rotate(270);    
  textSize(20);
  textFont(fontRegular);
  text("0 - 1", 0, 30);
  pop();
  push();
  textFont(fontBold);
  text("Government Social Media Monitoring", x, height - marginBtm + 50);
  pop();
  push();
  textSize(20);
  textFont(fontRegular);
  text("-4 - 4 Interval", x, height - marginBtm + 80);
  pop();
  push();
  textSize(16);
  textFont(fontItalics);
  text("Feldstein, Steven (2022), “AI & Big Data Global Surveillance Index (2022 updated)”, Mendeley Data, V4, doi: 10.17632/gjhf5y4xjp.4", x, height - marginBtm + 180);
  pop();
  
  //Draw key for military expenses when dots are drawn 
  push();
  expensesKey();
  pop();
  textAlign(CENTER); 
  textFont(fontRegular);

  //Draw coordinates
  drawCoordinates();

  pop();
}


function drawCountries(label, dotX, dotY){
  push();
  textAlign(CENTER, CENTER);
  fill(225);
  stroke(0);
  textSize(18);
  text(label, dotX, dotY); 
  pop();
}


function drawBars(){
  let numBars = data2.length; 
  let barWidth = 100; 
  let gap = 50; 
  let totalWidth = data2.getRowCount() * (barWidth + gap);
  let startX = (width - totalWidth) / 2;  
  for(let r = 0; r < data2.getRowCount(); r++){
    let rowObject = data2.getRow(r);
    let barX = startX + r * (barWidth + 30) + 40; 
    let barHeight = map(rowObject.getNum("billions"), minBil, maxBil, 50, height - marginBtm * 2);
    let barY = height - marginBtm - barHeight; 
    noStroke();
    fill(255);
    rect(barX, barY, barWidth, barHeight);
    push();
    fill(255);
    textAlign(CENTER, TOP);
    textSize(24);
    text(rowObject.getString("year"), barX + barWidth / 2, height - marginBtm + 10);
    textAlign(CENTER, BOTTOM); 
    text("$" + rowObject.getNum("billions") + " B", barX + barWidth / 2, barY - 5);
    pop();
  }
  push();
  fill(255);
  textSize(24);
  textAlign(CENTER);
  textFont(fontBold);
  text("AI in Surveillance Market Projection", x, height - marginBtm + 80);
  pop();
  push();
  textFont(fontRegular);
  textSize(20);
  text("in Billions", x, height - marginBtm + 110);
  pop();
  push();
  fill(255);
  textAlign(CENTER);
  textFont(fontItalics);
  textSize(16);
  text("Dhapte, A. (2026, April 6). AI in surveillance market size, share and research report (Report No. MRFR/ICT/37130-HCR). Market Research Future.", x, height - marginBtm + 180);
  pop();
}

function expensesKey(){
  noStroke();
  let steps = 40;
  let keyStartX = width - margin - 75;
  let keyEndX = width - margin + 100;
  let keyY = margin - 50;
  let keyHeight = 20;
  
  for (let i = 0; i < steps; i++) {
    let t = i / steps;
    let kx = lerp(keyStartX, keyEndX, t);
    let r = lerp(30, 50, t);
    fill(lerpColor(color(255, 255, 255, 0), color(255, 255, 255, 255), t));
    ellipse(kx, keyY, r);
  }

  fill(255);
  textSize(20);
  textFont(fontRegular);
  textAlign(CENTER);
  text("No data", keyStartX, keyY + 50);
  text("~$648.8K", keyEndX, keyY + 50);
  text("1", keyStartX, keyY - 50);
  text("4", keyEndX, keyY - 50);
  textFont(fontBold);
  text("Types of AI Used", (keyStartX + keyEndX) / 2, keyY - 100);
  text("Total Military Expenditures", (keyStartX + keyEndX) / 2, keyY + 100);
}


function context(){
  let x = width/2; //Center horizontally 
  let top = height - margin + 120; //Set top threshold 
  push();
  rectMode(CENTER);
  fill(0);
  textFont(fontMedItalics);
  rect(x, height, width, 200); //Draw black background section at bottom 
  fill(255, 255, 255, op); //Text color, opacity: https://editor.p5js.org/remarkability/sketches/rtM08miUD 
  textSize(24);
  textAlign(CENTER);
  if (op < 0){ 
    fadeAmount = 1.3; 
  }else if (op > 255){
    fadeAmount =- 3; 
  }
  op += fadeAmount; 
  text(lines[linesIndex], x, scrollY); //Draw line of text from text file  
  if(scrollY < top){ //If the text y-position reaches the top threshold, set opacity to 0
    linesIndex = (linesIndex + 1) % lines.length;
    scrollY = height - 10; //Reset y-position to bottom 
    op = 0; 
  }else{
    scrollY -= 0.25; //Y-pos of text animates up  
  }
  //Loop back to first line/index if lineIndex reaches the end 
  if(linesIndex == lines.length - 1){
    lineIndex = 0; 
  }
  pop();
}

function drawCoordinates(){
  push();
  textFont(fontItalics);
  textSize(16); 
  fill(255);
  let zeroX = map(0, minGov, maxGov, margin, width - margin);
  let zeroY = map(0, minDem, maxDem, height - marginBtm, margin);
  ellipse(zeroX, zeroY, 10); 
  stroke(255);
  strokeWeight(2);
  line(zeroX + 15, zeroY, zeroX - 15, zeroY); 
  line(zeroX, zeroY + 15, zeroX, zeroY - 15); 
  noStroke();
  text('(0,0)', zeroX, zeroY - 30); 
  stroke(255);
  pop();
}

function direcTxt(){
  push();
  textAlign(CENTER);
  textFont(fontBold);
  textSize(24);
  fill(255);
  text("Click anywhere or press any key to start audio", x, margin);
  pop();
}


//**VARIABLE DESCRIPTIONS**:
//V-Dem Index (v2x_polyarchy): Extent of ideal electoral democracy on a scale from 0 (low) to 1 (high). Includes free and fair elections, broad suffrage, freedom of expression, etc. 
//Global Social Media Surveillance (v2smgovsmmon): Comprehensiveness of surveillance in social media by the government or its agents, from none to extremely comprehensive (scored 0–4, converted to interval -4-4). Not exclusive to AI use. 
//Total Military Expenditures: reported military expenditures of each country in USD.
//Regime Types (v2x_regime): Classification based on liberal principles (civil liberties, rule of law, and executive constraints) and access to power (polyarchy). 
//Closed Autocracy: No multiparty elections for the chief executive or the legislature.
//Electoral Autocracy:  De-jure multiparty elections for the chief executive and the legislature but elections are not free and fair. 
//Electoral Democracy: De-facto free and fair multiparty elections and meets minimum level of Dahl’s institutional prerequisites, lacking in liberal principles. 
//Liberal Democracy: De-facto free and fair multiparty elections and minimum level of Dahl’s institutional prerequisites are gurantted, as well as liberal principles. 
//Types of AI Surveillance: 1 or more deployed by state authorities. Assumed to include non-private, but exclude Automated Border Control systems at airports. 
//Smart City/Safe City Platforms: Urban networks comprising thousands of sensors that transmit real-time data to facilitate city management with a public safety focus. 
//Facial Recognition Systems: Biometric technology that analyzes human faces for identification purposes. 
//Smart Policing: Data-driven methods for predictive policing, police response, investigations, and even sentencing decisions. 
//Social Media Surveillance: Algorithms that automatically monitor millions of online communications.
//V-Dem: https://www.v-dem.net/documents/55/codebook.pdf 
//Carnegie: https://www.ned.org/wp-content/uploads/2022/06/Global-Struggle-Over-AI-Surveillance-Emerging-Trends-Democratic-Responses.pdf


//**REFERENCES**: 
//Bounding box and splitting hand index: https://editor.p5js.org/CCMultiGroup7/sketches/aCZ-mKWif 
//Fading text: https://editor.p5js.org/remarkability/sketches/rtM08miUD 
//Vertical scrolling text: https://editor.p5js.org/codingtrain/sketches/1UggvR8ix 
//Loading CSV for NN: https://medium.com/aixdesign/getting-started-with-ml5-js-tutorial-part-iii-yoga-pose-detection-8b4609f122ab 
//External webcam setup: https://editor.p5js.org/lunahm/sketches/jaXBDF9Hu 
//Pointing Gesture Dataset Uploaded:  https://github.com/lunahmy/pointing-gesture-dataset
//Clear pose when hand leaves https://preview.p5js.org/21fa1cc2-eb59-4cd7-a6f6-da863d771c3f:154:9tile 0
//Font - Fir Neue: https://www.dafont.com/tt-firs-neue.font 
//Data Critique: https://ipvm.com/reports/carnegie 