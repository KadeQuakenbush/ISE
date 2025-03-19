# Lab1: Bug Report Classification 
## Dataset Details

The experiment uses bug report data from popular deep learning frameworks:
- PyTorch
- TensorFlow
- Keras
- Apache MXNet (incubator-mxnet)
- Caffe

### Data Structure
Each bug report contains:
- Repository: This refers to the name of the repository where the bug report was posted.
- Number: This represents the unique identification number of the specific bug report or issue within the repository.
- State: This indicates the current status or state of the issue. Common values include "open" and "closed".
- Title: This is the title or brief description of the issue or bug. It provides a quick summary of the problem reported. 
- Body: This column contains the detailed description of the issue, including steps to reproduce, error messages, and any additional context provided by the reporter.
- Labels: These are tags or categories that help classify the bug report based on its nature. Labels may include types of issues such as "bug", "enhancement", "documentation", etc.
- Comments: This contains any comments made by users or maintainers related to the issue, often used to clarify or discuss the problem further. 
- Codes: This include code snippets, error logs, or relevant technical details provided in the bug report that help in diagnosing the problem. 
- Commands: This could represent specific shell commands related to the bug report, such as command-line commands used to reproduce the issue or execute tests.
- Class: This column is used to categorize the bug report, indicating whether this issue is a bug report or not.)
