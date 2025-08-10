# 🤝 Contributing to Taxi Fare Prediction ML

Thank you for your interest in contributing to the Taxi Fare Prediction ML project! We welcome contributions from the community and are grateful for your support.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Standards](#development-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## 📜 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be collaborative**: Work together to improve the project
- **Be constructive**: Provide helpful feedback and suggestions
- **Be professional**: Maintain a professional tone in all interactions

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.8+** installed
- **Node.js 16+** and npm
- **Git** and **Git LFS** configured
- **Docker** and **Docker Compose** (optional but recommended)
- Basic knowledge of FastAPI, React, and TypeScript

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
```bash
git clone https://github.com/YOUR_USERNAME/taxi-fare-prediction-ml.git
cd taxi-fare-prediction-ml
```

3. **Add upstream remote**:
```bash
git remote add upstream https://github.com/moazmo/taxi-fare-prediction-ml.git
```

## 🛠️ Development Setup

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Environment Configuration

1. Copy environment templates:
```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

2. Update the `.env` files with your local configuration

### Running the Development Environment

```bash
# Option 1: Docker Compose (Recommended)
docker-compose up -d

# Option 2: Manual setup
# Terminal 1 - Backend
cd backend && uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend && npm start
```

## 📝 Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug fixes**: Fix issues and improve stability
- **✨ New features**: Add new functionality
- **📚 Documentation**: Improve or add documentation
- **🧪 Tests**: Add or improve test coverage
- **🎨 UI/UX**: Improve user interface and experience
- **⚡ Performance**: Optimize performance and efficiency
- **🔒 Security**: Enhance security measures

### Contribution Workflow

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for new features or significant changes
3. **Create a feature branch** from `main`
4. **Make your changes** following our coding standards
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request**

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Changes are focused and atomic

### Pull Request Template

When submitting a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks** must pass (linting, tests, builds)
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Approval** and merge by maintainers

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python/Node versions, etc.)
- **Screenshots** or error messages (if applicable)
- **Additional context** that might be helpful

### Feature Requests

For feature requests, please provide:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**
- **Additional context**

## 🎯 Development Standards

### Code Style

#### Python (Backend)
- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Maximum line length: **88 characters** (Black formatter)
- Use **docstrings** for all public functions and classes
- Import organization: standard library, third-party, local imports

#### TypeScript (Frontend)
- Follow **ESLint** configuration rules
- Use **TypeScript strict mode**
- Prefer **functional components** with hooks
- Use **meaningful variable names**
- Maximum line length: **100 characters**

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

Examples:
```
feat(api): add batch prediction endpoint
fix(frontend): resolve map rendering issue
docs(readme): update installation instructions
```

### Branch Naming

Use descriptive branch names:
- `feature/add-batch-predictions`
- `fix/map-rendering-issue`
- `docs/update-api-documentation`
- `refactor/improve-error-handling`

## 🧪 Testing

### Backend Testing

```bash
cd backend
python -m pytest tests/ -v --cov=app
python test_model.py
```

### Frontend Testing

```bash
cd frontend
npm test
npm run type-check
npm run lint
```

### Test Requirements

- **Unit tests** for new functions and classes
- **Integration tests** for API endpoints
- **Component tests** for React components
- **End-to-end tests** for critical user flows
- **Minimum 80% code coverage** for new code

## 📚 Documentation

### Documentation Standards

- **API documentation**: Update OpenAPI specs for API changes
- **Code comments**: Explain complex logic and algorithms
- **README updates**: Keep installation and usage instructions current
- **Inline documentation**: Use docstrings and JSDoc comments
- **Examples**: Provide usage examples for new features

### Documentation Locations

- **API docs**: `docs/API_DOCUMENTATION.md`
- **Deployment**: `docs/DEPLOYMENT_GUIDE.md`
- **Code docs**: Inline comments and docstrings
- **User guide**: Main `README.md`

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** acknowledgments section
- **CONTRIBUTORS.md** file (if created)
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📞 Getting Help

If you need help or have questions:

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For specific problems or bug reports
- **Code Review**: Ask for feedback during the PR process
- **Documentation**: Check existing docs and guides

## 🎉 Thank You!

Thank you for contributing to the Taxi Fare Prediction ML project! Your contributions help make this project better for everyone.

---

**Happy Contributing!** 🚀
