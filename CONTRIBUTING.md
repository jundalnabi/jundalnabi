# Contributing to PyQuotex AI Trading Bot

Thank you for your interest in contributing to PyQuotex! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of trading and machine learning concepts

### Setting Up Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/pyquotex.git
   cd pyquotex
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv dev_env
   dev_env\Scripts\activate  # Windows
   # or
   source dev_env/bin/activate  # Linux/Mac
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 How to Contribute

### Types of Contributions

1. **Bug Reports**: Report bugs and issues
2. **Feature Requests**: Suggest new features
3. **Code Contributions**: Fix bugs or add features
4. **Documentation**: Improve documentation
5. **Testing**: Add tests or improve existing ones

### Reporting Issues

When reporting issues, please include:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** or logs
- **Screenshots** if applicable

### Code Contributions

#### Code Style Guidelines

- Follow **PEP 8** Python style guide
- Use **type hints** for function parameters and return values
- Write **docstrings** for all functions and classes
- Use **meaningful variable names**
- Keep functions **small and focused**

#### Example Code Style

```python
def predict_direction(self, features: List[float]) -> Tuple[str, float]:
    """
    Predict trading direction using neural network.
    
    Args:
        features: List of market features
        
    Returns:
        Tuple of (direction, confidence)
    """
    if not features or len(features) != self.feature_size:
        return 'call', 0.5
    
    # Implementation here
    pass
```

#### Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: add neural network confidence scoring
fix: resolve memory leak in pattern learning
docs: update README with new commands
test: add unit tests for monitoring system
refactor: improve error handling in auto-trade
```

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Commit your changes** with clear messages
6. **Push to your fork** and create a Pull Request

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tested locally
- [ ] Added new tests
- [ ] All existing tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_neural_network.py

# Run with coverage
python -m pytest --cov=pyquotex
```

### Writing Tests

Create test files in the `tests/` directory:

```python
import pytest
from pyquotex.app import PyQuotexCLI

class TestNeuralNetwork:
    def test_model_creation(self):
        cli = PyQuotexCLI()
        cli.setup_model()
        assert cli.model is not None
        assert cli.feature_size == 36
    
    def test_prediction_confidence(self):
        cli = PyQuotexCLI()
        cli.setup_model()
        features = [0.5] * 36  # Mock features
        direction, confidence = cli.predict_with_confidence(features)
        assert direction in ['call', 'put']
        assert 0 <= confidence <= 1
```

## 📚 Documentation

### Documentation Standards

- **README.md**: Main project documentation
- **Code comments**: Explain complex logic
- **Docstrings**: Document all functions and classes
- **Type hints**: Use for better code understanding

### Updating Documentation

When adding new features:

1. **Update README.md** with new commands/features
2. **Add docstrings** to new functions
3. **Update help text** in command line interface
4. **Add examples** in documentation

## 🎯 Areas for Contribution

### High Priority
- **Bug fixes** and stability improvements
- **Performance optimizations**
- **Additional technical indicators**
- **More trading strategies**
- **Enhanced error handling**

### Medium Priority
- **Additional asset support**
- **Web interface** for monitoring
- **Mobile app** integration
- **Advanced analytics**
- **Risk management tools**

### Low Priority
- **UI/UX improvements**
- **Additional documentation**
- **Code refactoring**
- **Test coverage improvements**

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** in discussions

### Communication

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas
- **Pull Requests**: For code contributions

## 📋 Review Process

### For Contributors

1. **Self-review** your code before submitting
2. **Test thoroughly** on your local environment
3. **Follow guidelines** and best practices
4. **Respond to feedback** promptly

### For Maintainers

1. **Review code** for quality and correctness
2. **Test changes** in different environments
3. **Provide constructive feedback**
4. **Merge when ready** and appropriate

## 🏆 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📞 Getting Help

If you need help:

- **Check existing issues** first
- **Read documentation** thoroughly
- **Ask questions** in discussions
- **Contact maintainers** if needed

## 🎉 Thank You

Thank you for contributing to PyQuotex! Your contributions help make this project better for everyone.

---

**Happy coding! 🚀**
