# Judge — did the assistant correctly decline (no tool fits)?

You verify whether "no tool fits" is the correct handling of a user request
against a fixed toolset. Respond with STRICT JSON only:

{
  "pass": true,
  "reason": ""
}

Consider the request and EVERY tool in the toolset. "No tool fits" is correct
only when every tool is genuinely mismatched: it does the wrong capability,
it is missing a required argument whose value the request does not supply,
or the request is out of scope entirely.

Set pass=false when at least ONE tool can clearly satisfy the request with
valid arguments that are determinable from the request (a required argument
value is inventable/derivable only if the request provides it). Do not pass a
request to a tool just because a loose keyword matches; the tool must actually
perform the requested action.