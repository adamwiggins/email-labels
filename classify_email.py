"""
Classify an email (already converted to text or markdown) into one of three
categories using a language model.
"""

from providers import LLMProvider

def classify_email(provider: LLMProvider, content: str) -> str:
    prompt = f"""You are my executive assistant, and you are excellent at sorting through my emails and labeling them as Inbox, FYI, or Junk.

Inbox includes personal and professional correspondance with real humans that I know. It also may include automated emails from services I use when they require my action, for example login links. Also in inbox: investor updates, calendar invites. If they reference one of my projects such as The Browser Company, Muse, Ink & Switch, Heroku, or Local-First Conf then they usually go to the inbox.

FYI includes order receipts (for example, from Amazon) and newsletters I've subscribed to such as Money Stuff, Tangle, Benedict Evans, Hacker Newsletter, Elicit, Butter Docs, and Kevin Lynagh. Also included in FYI: security alerts, Patreon project updates, and Readwise highlights. All newsletters from buttondown.email are in FYI.

Junk is any sale or promotion (even from a service I've purchased from) and newsletters that I never subscribed to. All Substack newsletters go to junk (I read them in the app instead).

Response to the email below the line with just the label, nothing else.

The email to label is below.
---
"""
    return provider.get_completion(content, prompt)
