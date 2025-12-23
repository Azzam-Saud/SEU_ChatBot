<script>
function renderAnswer(text) {
  // تنظيف
  text = text.trim();

  // تقسيم الأسطر
  let lines = text.split("\n").map(l => l.trim()).filter(l => l);

  // لو فيه تعداد واضح
  const isList = lines.filter(l =>
    l.match(/^[-•\d]/) || l.includes("،")
  ).length > 3;

  // لو طويل جدًا → فقرة + تقسيم
  if (lines.join(" ").length > 600) {
    return lines.map(l => `<p>${l}</p>`).join("");
  }

  // لو تعداد
  if (isList) {
    const items = lines
      .join("،")
      .split("،")
      .map(i => i.trim())
      .filter(i => i.length > 3);

    return `
      <ul>
        ${items.map(i => `<li>${i}</li>`).join("")}
      </ul>
    `;
  }

  // الافتراضي
  return lines.map(l => `<p>${l}</p>`).join("");
}
</script>