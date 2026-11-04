# Slide 2: The Task and One Data Point

**Time:** 0:30

**Goal:** Remind the class what the task is and show one real example.

## On the slide

- The five tones: Appreciative or positive, Neutral or informational, Help-seeking or asking a question, Frustrated or complaining, Escalatory or requesting immediate resolution
- A training conversation (id 226352) shown as chat bubbles
- Its label: Frustrated or complaining

## What to say

> The data is Twitter conversations between customers and AmazonHelp, from the public Customer Support on Twitter dataset. The task is to label the tone of the customer's last message, using earlier turns as context. There are five tones, from appreciative to escalatory.
>
> Here is a real row from our training file. The customer pre-ordered FIFA 18, it hasn't arrived, and the last message says "won't be pre ordering with amazon again." That is a complaint, but there is no demand or threat, so it is Frustrated, not Escalatory.

## Key facts

- Source: Customer Support on Twitter (Kaggle), AmazonHelp conversations only
- Each row has `id`, `conversation`, `last_customer_message` and `label`
- When a message shows more than one tone, the order of precedence is Escalatory, Frustrated, Help-seeking, Appreciative, Neutral

## If asked

- **Why only the last message?** It is the customer's current state, which is what a support team would act on. Earlier turns help when the message is short or sarcastic.
- **Why is this example not Escalatory?** Escalatory needs a demand, a threat, an accusation or clearly intensified anger. "Won't be pre ordering again" is a complaint, not a demand.

[Previous](slide_01_title.md) | [Next](slide_03_annotation_process.md)
