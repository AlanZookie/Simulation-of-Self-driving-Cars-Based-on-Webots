// Copyright 1996-2020 Cyberbotics Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <QtCore/QSettings>
#include <webots/Joystick.hpp>
#include <webots/vehicle/Driver.hpp>
#include <webots/Camera.hpp>

#include "JoystickInterface.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#define NAXIS 3
#define NAXISBOUNDS 2
#define NBUTTONS 13
#define NGAINS 2

using namespace webots;
using namespace std;

static const char *axesNames[NAXIS] = {"Steering", "Throttle", "Brake"};

static const char *axesBoundNames[NAXISBOUNDS] = {"min", "max"};

static const char *buttonNames[NBUTTONS] = {"RightWarning", "LeftWarning",    "NextGear",          "PreviousGear", "FirstGear",
                                            "SecondGear",   "ThirdGear",      "FourthGear",        "FifthGear",    "SixthGear",
                                            "ReverseGear",  "NextWiperMode ", "PreviousWiperMode "};

static const char *gainNames[NGAINS] = {"AutoCentering", "Resistance"};

int pause_counter=0;
int image_num=0;

#ifdef _WIN32
static string platformExtension = "windows";
#elif defined(__APPLE__)
static string platformExtension = "mac";
#elif defined(__linux__)
static string platformExtension = "linux";
#else
#error Unsupported OS
#endif

void JoystickInterface::fatal(const string &txt) {
  cerr << txt << endl;
  exit(-1);
}

bool JoystickInterface::fileExists(const string &name) {
  ifstream f(name.c_str());
  return f.good();
}

double JoystickInterface::convertFeedback(int raw, int minimum, int maximum) {
  if (maximum == minimum)
    fatal("Prevent division by 0.");
  return std::max(0.0, std::min(1.0, ((double)(raw - minimum)) / ((double)(maximum - minimum))));
}

void JoystickInterface::init(webots::Driver *driver, const char *configFile) {
  mDriver = driver;

  mGear = 1;
  mWiperMode = Driver::DOWN;
  mDriver->setGear(mGear);
  mDriver->setWiperMode(mWiperMode);

  if (!mJoystick) {
    mJoystick = driver->getJoystick();
    mJoystick->enable(mDriver->getBasicTimeStep());
    driver->step();
  }

  if (mJoystick->isConnected())
    cout << "'" << mJoystick->getModel() << "' detected (the following configuration file is used: '" << configFile << "')."
         << endl;

  QSettings settings(configFile, QSettings::IniFormat);
  settings.setIniCodec("UTF-8");
  for (int i = 0; i < NAXIS; ++i)
    mAxesMap[axesNames[i]] = settings.value(QString("Axis/") + QString(axesNames[i]), -1).toInt();
  for (int i = 0; i < NAXIS; ++i) {
    for (int j = 0; j < NAXISBOUNDS; ++j) {
      string boundName = string() + axesBoundNames[j] + axesNames[i];
      mAxesBoundsMap[boundName] = settings.value(QString("AxisBounds/") + QString::fromStdString(boundName), 0).toInt();
    }
  }
  for (int i = 0; i < NBUTTONS; ++i)
    mButtonsMap[buttonNames[i]] = settings.value(QString("Buttons/") + QString(buttonNames[i]), -1).toInt();
  for (int i = 0; i < NGAINS; ++i)
    mGainMap[gainNames[i]] = settings.value(QString("Gains/") + QString(gainNames[i]), 0.0).toDouble();

  if (mJoystick->isConnected())
    mJoystick->setForceAxis(mAxesMap["Steering"]);
}

JoystickInterface::JoystickInterface(webots::Driver *driver) {

  int time_step = driver->getBasicTimeStep();
  // 1. find out "joystick_configuration_file" associated to the connected joystick
  string configurationFile;
  mJoystick = driver->getJoystick();
  mJoystick->enable(driver->getBasicTimeStep());
  //enable the cameras
  fCamera = driver->getCamera("front_camera");
  flCamera = driver->getCamera("frontleft_camera");
  frCamera = driver->getCamera("frontright_camera");
  fCamera->enable(driver->getBasicTimeStep());
  flCamera->enable(driver->getBasicTimeStep());
  frCamera->enable(driver->getBasicTimeStep());  
 
  driver->step();

  if (!mJoystick->isConnected())
    return;

  string model = mJoystick->getModel();
  if (model.find("G29") != std::string::npos)
    configurationFile = "config_logitech_g29.ini";
  else if (model.find("G27") != std::string::npos)
    configurationFile = "config_logitech_g27.ini";
  else if (model.find("PXN-V9") != std::string::npos)
    configurationFile = "config_V9.ini";
  else if (model.find("FANATEC CSL Elite Wheel Base") != std::string::npos)
    configurationFile = "config_fanatec_csl_elite_wheel_base.ini";
  else
    fatal("'" + model + "' not supported please provide a custom configuration file in argument.");

  // 2. look for platform specific configuration file:
  string platformConfigurationFile = configurationFile;
  platformConfigurationFile.erase(platformConfigurationFile.end() - 4, platformConfigurationFile.end());
  platformConfigurationFile += "." + platformExtension + ".ini";
  if (fileExists(platformConfigurationFile))
    configurationFile = platformConfigurationFile;

  init(driver, configurationFile.c_str());
}

JoystickInterface::JoystickInterface(webots::Driver *driver, const char *configFile) {
  mJoystick = NULL;
  if (!fileExists(configFile))
    fatal("File '" + string(configFile) + "' does not exist.");
  init(driver, configFile);
}

bool JoystickInterface::step() {
  if (!mJoystick->isConnected())
    return false;
    
    //picture
  char file_name[200];
  int time_step = mDriver->getBasicTimeStep();
  const unsigned char *image_f=fCamera->getImage();
 

  //useful debug code: display joystick state
  
  cout << "axes:" << endl;
  for (int i = 0; i < mJoystick->getNumberOfAxes(); ++i)
    cout << "- axe " << i << " " << mJoystick->getAxisValue(i) << endl;
  cout << "povs:" << endl;
  for (int i = 0; i < mJoystick->getNumberOfPovs(); ++i)
    cout << "- pov " << i << " " << mJoystick->getPovValue(i) << endl;
  int b = mJoystick->getPressedButton();
  cout << "buttons:" << endl;
  //while (b >= 0) {
    cout << b << " ";
    //b = mJoystick->getPressedButton();
 // }
  cout << endl << endl;
  

  // update steering, throttle, and brake based on axes value

  // raw data
  int steeringFeedback = mJoystick->getAxisValue(mAxesMap["Steering"]);
  int throttleFeedback = mJoystick->getAxisValue(mAxesMap["Throttle"]);
  int brakeFeedback = mJoystick->getAxisValue(mAxesMap["Brake"]);

  // bounded scaled data [0, 1]
  double steeringAngle = convertFeedback(steeringFeedback, mAxesBoundsMap["minSteering"], mAxesBoundsMap["maxSteering"]);
  double throttle = convertFeedback(throttleFeedback, mAxesBoundsMap["minThrottle"], mAxesBoundsMap["maxThrottle"]);
  double brake = convertFeedback(brakeFeedback, mAxesBoundsMap["minBrake"], mAxesBoundsMap["maxBrake"]);
  // useful debug code: display the resulting scaled data before sending it to the driver library
   cout << "steering:" << steeringAngle << " throttle:" << throttle << " brake:" << brake << endl;

  // to automobile API
  mDriver->setSteeringAngle(steeringAngle - 0.5);  // convert to [-0.5, 0.5] radian range
  mDriver->setThrottle(throttle);
  mDriver->setBrakeIntensity(brake);

  // update gear and indicator based on buttons state
  //int button = mJoystick->getPressedButton();
  //cout << button << " ";
  int gear = mGear;
  webots::Driver::WiperMode wiperMode = mWiperMode;
  static bool wasSwitchingToNextGear = false;
  static bool wasSwitchingToPreviousGear = false;
  bool isSwitchingToNextGear = false;
  bool isSwitchingToPreviousGear = false;
  static bool wasLeftBlinkerOn = false;
  static bool wasRightBlinkerOn = false;
  bool isLeftBlinkerOn = false;
  bool isRightBlinkerOn = false;
  static bool wasSwitchingToNextWiperMode = false;
  static bool wasSwitchingToPreviousWiperMode = false;
  bool isSwitchingToNextWiperMode = false;
  bool isSwitchingToPreviousWiperMode = false;

  while (b >= 0) {
    if (b == mButtonsMap["NextGear"]) {
      if (!wasSwitchingToNextGear)
        gear += 1;
      isSwitchingToNextGear = true;
    } else if (b == mButtonsMap["PreviousGear"]) {
      if (!wasSwitchingToPreviousGear)
        gear -= 1;
      isSwitchingToPreviousGear = true;
    } else if (b == mButtonsMap["FirstGear"])
      gear = 1;
    else if (b == mButtonsMap["SecondGear"])
      gear = 2;
    else if (b == mButtonsMap["ThirdGear"])
      gear = 3;
    else if (b == mButtonsMap["FourthGear"])
      gear = 4;
    else if (b == mButtonsMap["FifthGear"])
      gear = 5;
    else if (b == mButtonsMap["SixthGear"])
      gear = 6;
    else if (b == mButtonsMap["ReverseGear"])
      gear = -1;
    else if (b == mButtonsMap["RightWarning"]) {
      if (!wasRightBlinkerOn)  // not pressed previous step
        mDriver->getIndicator() == Driver::INDICATOR_RIGHT ? mDriver->setIndicator(Driver::INDICATOR_OFF) :
                                                             mDriver->setIndicator(Driver::INDICATOR_RIGHT);
      isRightBlinkerOn = true;
    } else if (b == mButtonsMap["LeftWarning"]) {
      if (!wasLeftBlinkerOn)  // not pressed previous step
        mDriver->getIndicator() == Driver::INDICATOR_LEFT ? mDriver->setIndicator(Driver::INDICATOR_OFF) :
                                                            mDriver->setIndicator(Driver::INDICATOR_LEFT);
      isLeftBlinkerOn = true;
    } else if (b == mButtonsMap["NextWiperMode "] || wiperMode < Driver::FAST) {
      if (!wasSwitchingToNextWiperMode)
        wiperMode = webots::Driver::WiperMode(wiperMode + 1);
      isSwitchingToNextWiperMode = true;
    } else if (b == mButtonsMap["PreviousWiperMode "] || wiperMode > Driver::DOWN) {
      if (!wasSwitchingToPreviousWiperMode)
        wiperMode = webots::Driver::WiperMode(wiperMode - 1);
      isSwitchingToPreviousWiperMode = true;
    }

    b = mJoystick->getPressedButton();
  }
  wasSwitchingToNextGear = isSwitchingToNextGear;
  wasSwitchingToPreviousGear = isSwitchingToPreviousGear;
  wasLeftBlinkerOn = isLeftBlinkerOn;
  wasRightBlinkerOn = isRightBlinkerOn;
  wasSwitchingToNextWiperMode = isSwitchingToNextWiperMode;
  wasSwitchingToPreviousWiperMode = isSwitchingToPreviousWiperMode;
  
  cout << "gear: " << gear << endl;
  //gear = std::max(-1, std::min(mDriver->getGearNumber(), gear));
  //if (gear != mGear) {
    mGear = gear;
    mDriver->setGear(mGear);
  //}
  cout << "gear: " << mDriver->getGear()<< endl;
  
  if (wiperMode != mWiperMode) {
    mWiperMode = wiperMode;
    mDriver->setWiperMode(mWiperMode);
  }

  // update resistance and auto-centering gain based on speed
  static const double maxSpeed = 40.0; // speed from which the max gain is applied
  double speed = mDriver->getCurrentSpeed();
  cout<<"The speed is "<<speed<<endl;
  if (speed > maxSpeed)
    mDriver->setCruisingSpeed(maxSpeed);
  if (mGainMap["AutoCentering"] > 0.0)
    mJoystick->setAutoCenteringGain(speed > maxSpeed ? mGainMap["AutoCentering"] :
                                                       mGainMap["AutoCentering"] * speed / maxSpeed);
  if (mGainMap["Resistance"] > 0.0)
    mJoystick->setResistanceGain(speed > maxSpeed ? 0.0 : mGainMap["Resistance"] * (1.0 - speed / maxSpeed));
    
    
  //try to save the images for every 2 seconds
  if (pause_counter>0){
    cout<<pause_counter<<endl;
    pause_counter--;
  }
  else{
  //save  image
    sprintf(file_name,"D:\\_1Asenior\\gra_pro\\four\\photo\\fphoto_%d.png",image_num);
    if(fCamera->saveImage(file_name,100)!=0){
    cout<<"photo save error"<<endl;
  //create .csv
 
    } 
    ofstream tb;
    tb.open("D:\\_1Asenior\\gra_pro\\four\\driving_log1.csv",ios::out|ios::app);
    tb<<file_name<<','<<file_name<<','<<file_name<<','<<steeringAngle - 0.5<<','<<throttle<<','<<brake<<','<<speed<<endl;
    tb.close();
    image_num++;
    pause_counter=80/time_step;
  }  
  //

  return true;
}
