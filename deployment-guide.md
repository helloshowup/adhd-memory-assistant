# Deployment Guide

## Local Deployment

### Prerequisites
- Python 3.8+
- Git
- Anthropic API key

### Step-by-Step Setup

1. **Environment Setup**
```bash
# Clone repository
git clone [repository-url]
cd adhd-memory-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file
ANTHROPIC_API_KEY=your_key_here
```

3. **Run Application**
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. **Preparation**
- Fork repository to your GitHub
- Get Anthropic API key
- Have your repository public or use Streamlit for Teams

2. **Deployment Steps**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub account
- Select repository and branch
- Add environment variables:
  - ANTHROPIC_API_KEY

3. **Post-Deployment**
- Monitor logs for errors
- Check ChromaDB persistence
- Verify API connectivity

## Maintenance

### Updates
```bash
# Update dependencies
pip install -U -r requirements.txt

# Update API keys if needed
vi .env
```

### Monitoring
- Check Streamlit logs
- Monitor API usage
- Review ChromaDB storage

### Troubleshooting
- Verify API key
- Check environment variables
- Review connection errors
- Clear ChromaDB cache if needed