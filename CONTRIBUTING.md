# Contributing to CUDA Programming Tutorial

Thank you for your interest in improving this tutorial! Contributions from students, engineers, and educators make this resource better for everyone learning GPU programming.

---

## Ways to Contribute

| Type | Examples |
| :--- | :--- |
| **Content** | Fix typos, clarify explanations, add diagrams, write new modules |
| **Code** | Runnable examples in `examples/`, exercises, benchmark fixes |
| **Projects** | New end-to-end projects in `projects/` |
| **Documentation** | Cheatsheets, glossary entries, setup guides |
| **Issues** | Report bugs, outdated CUDA versions, broken links |

---

## Before You Start

1. **Read the roadmap** in [README.md](README.md) to see which modules are planned vs. complete.
2. **Check existing issues and pull requests** to avoid duplicate work.
3. **Follow the module order** — new lecture content should fit the learning path (foundations → core → intermediate → advanced).

---

## Development Setup

### Requirements

- NVIDIA GPU with a supported driver
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.x recommended)
- `nvcc` available on your PATH
- Git

### Verify your environment

```bash
nvidia-smi
nvcc --version
```

### Compile an example

```bash
cd examples/thread-indexing
nvcc indexing.cu -o indexing && ./indexing
```

---

## Repository Structure

```
lectures/          ← Primary tutorial content (one README per module)
examples/          ← Small, self-contained .cu snippets
exercises/         ← Practice problems
projects/          ← Larger buildable projects
quizzes/           ← Self-assessment per module
benchmarks/        ← CPU vs GPU comparisons
cheatsheets/       ← Quick-reference cards
resources/         ← Curated external links
diagrams/          ← Architecture visuals
docs/              ← Extended documentation
```

When adding content, place files in the folder that matches their purpose.

---

## Content Guidelines

### Lecture modules (`lectures/module-XX-*/README.md`)

Each module README should include:

- **Header** — estimated time, prerequisites, link to next module
- **What You Will Learn** — bullet list of outcomes
- **Numbered parts** — logical sections with code examples
- **Summary table** — key concepts at the end
- **External references** — links to official NVIDIA documentation

### Writing style

- Use clear, beginner-friendly language; explain jargon on first use.
- Prefer **runnable code** over pseudocode.
- Include **why** something matters for performance, not only syntax.
- Use `> [!TIP]`, `> [!IMPORTANT]`, and `> [!NOTE]` callouts for critical points (same as existing modules).
- Keep code comments concise and educational.

### Code style

- CUDA C++ with `__global__`, `__device__`, `__host__` used correctly.
- Always show bounds checks: `if (i < N)` where appropriate.
- Include `CUDA_CHECK` or equivalent error handling in host code when showing full programs.
- Block sizes should be multiples of 32 (warp size) unless demonstrating a specific pitfall.
- File names: lowercase with hyphens (e.g. `memory-coalescing.cu`).

---

## Pull Request Process

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/module-06-examples
   ```

2. **Make focused changes** — one topic or fix per PR when possible.

3. **Test code** — compile and run any `.cu` files you add or modify.

4. **Update links** — if you add a module, update [README.md](README.md) module index and status.

5. **Open a pull request** with:
   - A clear title (e.g. `Add Module 06 memory coalescing examples`)
   - Summary of what changed and why
   - How you tested (commands run, GPU used if relevant)

6. **Respond to review feedback** — maintainers may request edits for clarity or consistency.

---

## Reporting Issues

When filing an issue, please include:

- Module or file path (e.g. `lectures/module-05-threads-blocks-grids/README.md`)
- CUDA Toolkit version (`nvcc --version`)
- GPU model (`nvidia-smi`)
- Expected vs. actual behavior
- Minimal reproduction steps for code bugs

---

## Code of Conduct

- Be respectful and constructive in issues, PRs, and discussions.
- Welcome learners at all skill levels.
- Give credit when adapting material from papers, books, or other tutorials.

---

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.

---

## Questions?

Open a [GitHub Issue](https://github.com/your-org/CUDA-programming-tutorial/issues) for questions about scope, module ideas, or technical direction. For small fixes, a pull request with a short description is often fastest.

**Thank you for helping others learn CUDA!**
