if (document.location.search.includes("dark-theme=true")) {
  document.body.classList.add("dark-theme");
}

let cursor = 0;
const RANGE = 5;
const LIMIT = 16_000;

const translateText = async (text) => {
  const inferResponse = await fetch(`infer_t5?input=${text}`);
  const inferJson = await inferResponse.json();

  return inferJson.output;
};

const textGenForm = document.querySelector(".text-gen-form");

textGenForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const textGenInput = document.getElementById("text-gen-input");
  const textGenParagraph = document.querySelector(".text-gen-output");

  try {
    textGenParagraph.textContent = await translateText(textGenInput.value);
  } catch (err) {
    console.error(err);
  }
});
