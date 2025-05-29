const evtSource = new EventSource("/stream");
evtSource.onmessage = function(event) {
  const alertDiv = document.getElementById("alert");
  const msg = document.createElement("p");
  msg.textContent = event.data;
  alertDiv.appendChild(msg);
};
