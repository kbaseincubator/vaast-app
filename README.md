# VAAST by KBase

VAAST is a web application for exploring genetic data and taxonomy using interactive visualizations and an LLM-powered chatbot interface.

## Installation

1.  **Clone or Pull the Repository**
    Ensure you have the latest version of the code:
    ```bash
    git pull origin main
    ```

2.  **Install Dependencies**
    This project uses [Poetry](https://python-poetry.org/) for dependency management. Install the dependencies with:
    ```bash
    poetry install
    ```

## Data Setup

The application requires specific data files that are too large to be stored in version control.

1.  **Download Data Files**
    You need to obtain the following pickle (`.pkl`) files from **FigShare**:
    *   `anthropic-docs-all.pkl`
    *   `openai-docs-all.pkl`

2.  **Place Files**
    Move these two files into the `data/` directory within the project root:
    ```
    vaast-app/
    └── data/
        ├── anthropic-docs-all.pkl
        └── openai-docs-all.pkl
    ```

## Starting the App

To start the application, rely on Poetry to handle the environment:

```bash
poetry run python -m vaast_app.app
```

Alternatively, if you have activated the poetry shell (`poetry shell`), you can run:

```bash
python -m vaast_app.app
```

The application will start on `http://0.0.0.0:8000` (accessible via `http://localhost:8000`).

## Configuration

You can configure the application using environment variables or through the Settings panel in the user interface.

### Environment Variables

The following environment variables can be set:

*   `OPENAI_API_KEY`: API key for OpenAI (required if using OpenAI models).
*   `ANTHROPIC_API_KEY`: API key for Anthropic (required if using Anthropic models).
*   `VERSION`: Specifies the backend provider version (e.g., `OpenAI`, `Anthropic`).
*   `HOSTING_LOCATION`: Specifies the hosting location (e.g., `API`, `CBORG`).
*   `MODEL`: Specifies the model to use.

You can set these in your shell before running the application:

```bash
export OPENAI_API_KEY="your-api-key"
export VERSION="OpenAI"
poetry run python -m vaast_app.app
```

### Settings Panel

Alternatively, you can configure these settings within the application:

1.  Start the app and open it in your browser.
2.  Click the **Settings** button in the top right corner.
3.  Enter the values for the relevant keys and configuration options.
4.  Click **Save**. The application will potentially reload updated documentation data if the Version is changed.

## Using the App

Once the application is running, open your web browser and navigate to:
[http://localhost:8000](http://localhost:8000)

The application consists of:
*   **Landing Page**: The entry point for the application.
*   **Visualizer**: Explore genetic data and taxonomy trees interactively.
*   **Chatbot**: Use the integrated chatbot to ask questions about the data, powered by LLM contexts loaded from the data files.
