function toggleMenu() {
        const nav = document.getElementById("navLinks");
        nav.classList.toggle("active");
      }

      const phrases = [
        "Crafted with ❤️ by Srashti Shukla",
        "AI that cares for your health",
        "Predict blood pressure. Prevent risk.",
        "Your health, our priority.",
      ];

      let currentPhrase = 0;
      let currentChar = 0;
      const typingSpeed = 100;
      const erasingSpeed = 50;
      const delayBetween = 1500; // Delay before erasing
      const typedText = document.getElementById("typed-text");

      function typePhrase() {
        if (typedText) {
          if (currentChar < phrases[currentPhrase].length) {
            typedText.textContent += phrases[currentPhrase].charAt(currentChar);
            currentChar++;
            setTimeout(typePhrase, typingSpeed);
          } else {
            setTimeout(erasePhrase, delayBetween);
          }
        }
      }

      function erasePhrase() {
        if (typedText) {
          if (currentChar > 0) {
            typedText.textContent = phrases[currentPhrase].substring(
              0,
              currentChar - 1
            );
            currentChar--;
            setTimeout(erasePhrase, erasingSpeed);
          } else {
            currentPhrase = (currentPhrase + 1) % phrases.length;
            setTimeout(typePhrase, 500);
          }
        }
      }

      window.onload = () => {
        typePhrase();
      };

      window.addEventListener("load", function () {
        if (window.location.hash === "#contact") {
          history.replaceState(null, null, window.location.pathname);
          window.scrollTo(0, 0); // Scroll to top
        }
      });