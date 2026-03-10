-- Reset valid_email when email is changed
CREATE TRIGGER reset_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
SET NEW.valid_email =
IF(NEW.email <> OLD.email, 0, NEW.valid_email);
