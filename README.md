# CSE 406 - Security Sessional Course

**Level 4, Term 1**  
**Student ID: 2005110**

This repository contains the offline assignments for the CSE 406 Security Sessional Course, focusing on cryptographic implementations and security analysis.

## 📁 Project Structure

```
CSE406/
├── OFFLINE1/                    # AES Implementation Assignment
│   ├── 2005110/                 # Student implementation
│   │   ├── _2005110_aes.py      # Main AES encryption/decryption
│   │   ├── _2005110_alice.py    # Alice's communication module
│   │   ├── _2005110_bob.py      # Bob's communication module
│   │   ├── _2005110_ecc.py      # Elliptic Curve Cryptography
│   │   ├── info.py              # AES constants (S-boxes, mixers)
│   │   └── roundkey.py          # Key expansion implementation
│   └── Resources/               # Assignment materials
│       ├── AES-simulation.pdf
│       ├── CSE406 - Assignment 1.pdf
│       └── Lecture materials
└── OFFLINE2/                    # Website Fingerprinting Assignment
    ├── 2005110/                 # Student implementation
    │   ├── app.py               # Flask web application
    │   ├── collect.py           # Automated data collection
    │   ├── train.py             # Machine learning models
    │   ├── database.py          # Database management
    │   ├── dataset.json         # Collected fingerprinting data
    │   ├── requirements.txt     # Python dependencies
    │   ├── static/              # Web interface files
    │   │   ├── index.html       # Main interface
    │   │   ├── index.js         # Frontend logic
    │   │   ├── worker.js        # Web worker for timing
    │   │   └── warmup.js        # Warmup scripts
    │   ├── report/              # Assignment report
    │   │   └── 2005110.pdf
    │   └── saved_models/        # Trained ML models
    └── Resources/               # Assignment materials
        ├── specification.pdf
        └── template/            # Reference implementation
```

## 🚀 OFFLINE 1: AES Implementation

### Overview
Implementation of the Advanced Encryption Standard (AES) algorithm with support for:
- AES-128 encryption and decryption
- CBC (Cipher Block Chaining) mode of operation
- PKCS7 padding
- Key expansion and round key generation
- Communication simulation between Alice and Bob

### Key Features
- **Complete AES Implementation**: All AES operations including SubBytes, ShiftRows, MixColumns, and AddRoundKey
- **CBC Mode Support**: Implements Cipher Block Chaining for secure encryption
- **Communication Protocol**: Simulates secure communication between two parties
- **BitVector Integration**: Uses BitVector library for efficient bit manipulation

### Files Description
- `_2005110_aes.py`: Main AES implementation with encryption/decryption functions
- `_2005110_alice.py`: Alice's side of the communication protocol
- `_2005110_bob.py`: Bob's side of the communication protocol
- `_2005110_ecc.py`: Elliptic Curve Cryptography implementation
- `info.py`: AES constants including S-boxes and mixers
- `roundkey.py`: Key expansion algorithm implementation

### Usage
```bash
cd OFFLINE1/2005110/
python _2005110_aes.py
```

## 🔍 OFFLINE 2: Website Fingerprinting

### Overview
A comprehensive website fingerprinting system that demonstrates how timing attacks can be used to identify websites based on user interactions. The system includes:

- **Data Collection**: Automated collection of timing traces from websites
- **Machine Learning Models**: Neural networks for website classification
- **Web Interface**: Real-time visualization and analysis
- **Database Management**: Persistent storage of collected data

### Key Features
- **Automated Data Collection**: Selenium-based automation for collecting timing traces
- **Neural Network Models**: Both simple and complex CNN architectures
- **Real-time Visualization**: Heatmap generation and statistical analysis
- **Multi-website Support**: Configurable target websites
- **Persistent Storage**: SQLite database with JSON export capability

### Technology Stack
- **Backend**: Flask web framework
- **Frontend**: HTML5, JavaScript with Web Workers
- **Machine Learning**: PyTorch with CNN architectures
- **Automation**: Selenium WebDriver
- **Database**: SQLAlchemy with SQLite
- **Visualization**: Matplotlib and Seaborn

### Files Description
- `app.py`: Flask web application with API endpoints
- `collect.py`: Automated data collection using Selenium
- `train.py`: Machine learning model training and evaluation
- `database.py`: Database management and data persistence
- `static/`: Web interface files for real-time interaction
- `requirements.txt`: Python dependencies

### Installation & Setup

1. **Install Dependencies**:
```bash
cd OFFLINE2/2005110/
pip install -r requirements.txt
```

2. **Start the Web Application**:
```bash
python app.py
```

3. **Run Data Collection** (in a separate terminal):
```bash
python collect.py
```

4. **Train Models**:
```bash
python train.py
```

### Configuration
- **Target Websites**: Modify `WEBSITES` list in `collect.py`
- **Collection Parameters**: Adjust `TRACES_PER_SITE`, `INTERACTION_TIME` in `collect.py`
- **Model Parameters**: Configure `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` in `train.py`

### Web Interface
Access the web interface at `http://localhost:5000` to:
- View real-time timing traces
- Generate heatmap visualizations
- Analyze statistical data
- Download collected datasets

## 📊 Results & Analysis

### OFFLINE 1 Results
- Successfully implemented AES-128 encryption/decryption
- CBC mode operation with proper padding
- Secure communication protocol between parties
- Comprehensive testing with various input scenarios

### OFFLINE 2 Results
- **Data Collection**: Automated collection of timing traces from multiple websites
- **Model Performance**: 
  - Simple Model: Baseline classification performance
  - Complex Model: Enhanced accuracy with batch normalization
- **Visualization**: Real-time heatmap generation and statistical analysis
- **Database**: Persistent storage with JSON export functionality

## 🔧 Technical Details

### AES Implementation
- **Key Size**: 128-bit (AES-128)
- **Block Size**: 128-bit
- **Rounds**: 10 rounds
- **Modes**: CBC (Cipher Block Chaining)
- **Padding**: PKCS7

### Website Fingerprinting
- **Input Size**: 188 timing samples per trace
- **Architecture**: 1D Convolutional Neural Networks
- **Models**: Simple CNN and Complex CNN with batch normalization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Evaluation**: Confusion matrices and classification reports

## 📚 Dependencies

### OFFLINE 1
- `BitVector`: For efficient bit manipulation

### OFFLINE 2
- `flask==2.3.3`: Web framework
- `torch>=2.0.0`: Deep learning framework
- `selenium==4.15.0`: Web automation
- `numpy==1.24.4`: Numerical computing
- `matplotlib==3.7.2`: Plotting and visualization
- `scikit-learn>=1.0.0`: Machine learning utilities
- `sqlalchemy>=2.0.0`: Database ORM

## 🎯 Learning Objectives

### OFFLINE 1
- Understanding AES algorithm internals
- Implementation of cryptographic primitives
- Secure communication protocols
- Block cipher modes of operation

### OFFLINE 2
- Website fingerprinting techniques
- Timing attack analysis
- Machine learning in security
- Web application security
- Data collection and analysis

## 📝 Reports

Detailed reports for each offline assignment are available in:
- `OFFLINE1/Resources/`: Assignment specifications and materials
- `OFFLINE2/2005110/report/2005110.pdf`: Comprehensive analysis and results

## 🤝 Contributing

This is an academic project for CSE 406 Security Sessional Course. The implementations are designed for educational purposes and demonstrate various security concepts and cryptographic techniques.

## 📄 License

This project is part of academic coursework and is intended for educational purposes only.

---

**Student**: 2005110  
**Course**: CSE 406 - Security Sessional  
**Level**: 4, Term 1  
**Institution**: BUET 
