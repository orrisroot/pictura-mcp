# Agent skills (for people using Pictura MCP)

This directory contains a ready-to-use **Agent Skills** package (per the
[Agent Skills standard](https://agentskills.io/specification)) that teaches an
AI coding agent how to use the Pictura MCP server well:

```
skills/
└── pictura-mcp/
    └── SKILL.md
```

It is optional — the server's tools are self-describing — but it encodes the
operating policy (save returned images yourself, verify allowlisted ids, family
limits, prompting conventions, privacy) so any agent produces consistent
results.

## Install for your harness

Copy the `pictura-mcp` folder into your agent's skills directory, then trust/start
the agent.

| Harness | Skill directory |
|---|---|
| pi | `~/.pi/agent/skills/` or `~/.agents/skills/` |
| Claude Code | `~/.claude/skills/` |
| OpenAI Codex | `~/.codex/skills/` |
| Any (project) | `.agents/skills/` inside a project (after trust) |

Example:

```bash
# pi
mkdir -p ~/.pi/agent/skills && cp -r skills/pictura-mcp ~/.pi/agent/skills/

# Claude Code
mkdir -p ~/.claude/skills && cp -r skills/pictura-mcp ~/.claude/skills/
```

Then just describe what you want ("generate a cat picture", "make this image
pop-art keeping the same pose") and let the agent load the skill.
