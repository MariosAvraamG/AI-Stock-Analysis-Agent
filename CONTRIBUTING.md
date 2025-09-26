# Contributing to AI Stock Analysis Platform

Thank you for your interest in contributing! This document provides guidelines for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Provide detailed information about the problem
3. Include steps to reproduce the issue

### Suggesting Enhancements

For feature requests:
1. Check if the feature has already been requested
2. Provide a clear description of the proposed feature
3. Explain the use case and benefits

## 🚀 Development Setup

### Prerequisites

- Node.js 18+
- Python 3.9+
- Git
- Supabase account
- OpenAI API key

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ai-stock-analysis-platform.git
   cd ai-stock-analysis-platform
   ```

2. **Set up the backend**
   ```bash
   cd ai-stock-backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd ai-stock-frontend
   npm install
   ```

4. **Configure environment variables**
   - Copy `.env.example` to `.env` in both frontend and backend
   - Fill in your actual values

## 📝 Code Style Guidelines

### Frontend (TypeScript/React)

- Use TypeScript for type safety
- Follow React best practices
- Use functional components with hooks
- Implement proper error handling
- Write meaningful component and function names

### Backend (Python/FastAPI)

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Implement proper error handling
- Use async/await for database operations

### General Guidelines

- Write clear, self-documenting code
- Add comments for complex logic
- Use meaningful variable and function names
- Keep functions small and focused
- Write tests for new functionality

## 🧪 Testing

### Frontend Testing
```bash
cd ai-stock-frontend
npm run test
```

### Backend Testing
```bash
cd ai-stock-backend
python -m pytest
```

## 📋 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   - Run all tests
   - Test manually in development environment
   - Ensure no breaking changes

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

## 🔍 Code Review Process

All pull requests will be reviewed for:
- Code quality and style
- Functionality and correctness
- Test coverage
- Documentation updates
- Security considerations

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, browser, versions)
- Error messages or logs

## 💡 Feature Requests

For feature requests, please provide:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach
- Impact on existing functionality

## 🛡️ Security

- Never commit sensitive information
- Use environment variables for secrets
- Follow security best practices
- Report security vulnerabilities privately

## 📞 Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for urgent issues

## 🎉 Recognition

Contributors will be recognized in:
- README contributors section
- Release notes
- Project documentation

Thank you for contributing to the AI Stock Analysis Platform! 🚀