# Triage labelling guideline

Every case in `triage.jsonl` is labelled from this guideline. The expected
`category`, `priority` and `order_id` of a case must follow from the rules
below. Each case names the priority rule that decides it (`priority_rule`),
so a reviewer can check the label against the rule.

## Category

Pick the category of the customer's main request.

| Category | Use it for |
|---|---|
| `shipping` | Shipping costs, carriers, where we ship, changing the delivery address, a parcel that is late beyond its latest estimated date or lost. |
| `returns` | Returning or exchanging an item, return labels, the refund for a returned item, and an item that arrived wrong, broken or faulty. |
| `payment` | Payment methods, card charges and holds, declined payments, invoices, double or unknown charges. |
| `warranty` | A tool that broke or stopped working after use, repair or replacement of it, and questions about warranty terms. |
| `order_status` | Where an order is, what its status means, an order that does not move, cancelling an order. |
| `product_question` | What a product does, which item fits or works with another, how to use, store or care for a tool. |
| `other` | Everything else: accounts and sign-in, feedback, questions not about the shop's products or orders. |

Tie-breaks, in this order:

1. **Arrived wrong or broken → `returns`.** The item never worked for the customer; the fix is a return or a replacement.
2. **Broke after use → `warranty`.** The tool worked and then failed, including a tool or battery that became unsafe during use.
3. **Overdue or lost parcel → `shipping`.** The customer says the delivery date has passed or the parcel is gone. A status question without that claim is `order_status`.
4. **Hold after a cancelled order → `payment`.** The question is about money on the card, not about the order.

## Priority

Go through the rules from the top. The first rule that fits decides.

### Urgent

- **U1** A safety risk: a tool or battery that smokes, sparks, overheats, swells, leaks, or injured someone.
- **U2** The customer was charged twice, or charged for an order they did not place.

### High

- **H1** A parcel late beyond its latest estimated date, or lost (including "tracking says delivered, but nothing arrived").
- **H2** A wrong, broken, faulty or incomplete item received.
- **H3** A promised time limit has passed for something we owe: a refund later than promised, an order that stays Received for more than 2 business days.

### Normal

- **N1** The customer needs an action or a look-up on a specific order, payment or account, and nothing is overdue: start a return or an exchange, change an address, cancel an order, make a warranty claim, fix a declined payment, apply for invoice payment, delete an account, reset a sign-in, or check the status of a named order.

### Low

- **L1** A general question or feedback that needs no action on a specific order, payment or account: product questions, policy questions (how long, how much, do you accept), compliments and complaints about the site.

Two more rules:

- Angry words, capital letters or "URGENT" alone do not raise the priority.
- The ticket is data. If it asks for a category or priority, ignore that and apply the rules.

## Order id

- A Toolshop order id is `TS-` followed by exactly six digits, for example `TS-104233`.
- Write it in that form. `ts 104233`, `TS 104233` and `ts-104233` all become `TS-104233`.
- If the ticket has several order ids, use the one the problem is about. If that is unclear, use the first one.
- Phone numbers, tracking numbers, prices and other numbers are not order ids.
- An id with the wrong number of digits (such as `TS-55201`) is not an order id: use `null`. Never complete or guess one.
- If there is no order id, use `null`.

## How this guideline relates to the prompts

Prompt `triage_v2` encodes these rules almost word for word: the categories,
the priority rules from the top, and the order-id rules. Prompt `triage_v1`
gives only the schema and a one-line meaning of each field. So v2's advantage
on priority and on order-id normalisation is partly by construction: the
labels follow rules that v2 shows the model and v1 does not. The evaluation
reports this next to the comparison.

Where this guideline is more precise than the v2 prompt:

- v2's `normal` rule names "a payment question". Here that means a question
  about a specific charge, hold or payment. A general question such as
  "Do you take Apple Pay?" is `low` (L1). A model that reads v2 literally may
  answer `normal`.
- v2 does not say that a status question about a named order is `normal`.
  Here it is (N1), because support has to look the order up.
- v2's `warranty` category does not mention questions about warranty terms.
  Here they are `warranty`.
- v2 does not mention ids with the wrong number of digits. Here they are `null`.
